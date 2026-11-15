// dump_reference.cpp -- observational reference dumper for MESHWEAVE Phase 3A.1
//
// Runs the exact GGUF through llama.cpp and writes intermediate tensors to disk so
// the custom runtime can be compared against the real reference rather than against
// guessed values.
//
// Captured per run:
//   logits.bin            float32 [n_vocab]      logits at the last token
//   final_hidden.bin      float32 [seq*n_embd]   hidden after the last layer + final norm
//   layerNN_input.bin     float32 [seq*n_embd]   input to layer NN (i.e. output of NN-1)
//
// layer00_input is the state right after token embedding, which is the first thing
// the custom runtime can be checked against.
//
// This performs no model mathematics. It only reads buffers llama.cpp already
// produces, and only in this diagnostic binary.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool write_floats(const std::string & path, const float * data, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    size_t written = fwrite(data, sizeof(float), n, f);
    fclose(f);
    return written == n;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <outdir> <token_id> [token_id ...]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const std::string outdir = argv[2];

    std::vector<llama_token> tokens;
    for (int i = 3; i < argc; ++i) {
        tokens.push_back((llama_token) atoi(argv[i]));
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    const int n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_embd   = llama_model_n_embd(model);
    const int n_layers = llama_model_n_layer(model);

    fprintf(stderr, "n_vocab=%d n_embd=%d n_layers=%d n_tokens=%zu\n",
            n_vocab, n_embd, n_layers, tokens.size());

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    cparams.embeddings = true;   // final hidden state
    cparams.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    // Which layers to instrument. Enabling all of them at once aborts inside
    // build_graph (the graph cannot hold that many extra outputs), so the set is
    // chosen by the caller: MESHWEAVE_DUMP_LAYERS="0,1,2" or "none".
    std::vector<int> want_layers;
    {
        const char * env = getenv("MESHWEAVE_DUMP_LAYERS");
        if (env && strcmp(env, "none") != 0) {
            std::string spec(env);
            size_t start = 0;
            while (start < spec.size()) {
                size_t comma = spec.find(',', start);
                if (comma == std::string::npos) comma = spec.size();
                int lid = atoi(spec.substr(start, comma - start).c_str());
                if (lid >= 0 && lid < n_layers) want_layers.push_back(lid);
                start = comma + 1;
            }
        }
    }
    fprintf(stderr, "instrumenting %zu layer(s)\n", want_layers.size());
    for (int lid : want_layers) {
        llama_set_embeddings_layer_inp(ctx, (uint32_t) lid, true);
    }

    llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (i == tokens.size() - 1) ? 1 : 0;
    }
    batch.n_tokens = (int32_t) tokens.size();

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    const int seq = (int) tokens.size();
    int failures = 0;

    // Logits at the final position.
    const float * logits = llama_get_logits_ith(ctx, (int32_t) tokens.size() - 1);
    if (logits) {
        if (!write_floats(outdir + "/logits.bin", logits, (size_t) n_vocab)) failures++;
    } else {
        fprintf(stderr, "no logits\n"); failures++;
    }

    // Final hidden state (after the last layer and final norm).
    const float * emb = llama_get_embeddings_ith(ctx, (int32_t) tokens.size() - 1);
    if (emb) {
        if (!write_floats(outdir + "/final_hidden.bin", emb, (size_t) n_embd)) failures++;
    } else {
        fprintf(stderr, "no final embeddings\n"); failures++;
    }

    // Per-layer inputs: layer L input is the output of layer L-1.
    for (int lid : want_layers) {
        const float * lin = llama_get_embeddings_layer_inp(ctx, (uint32_t) lid);
        if (!lin) { fprintf(stderr, "no layer_inp for %d\n", lid); failures++; continue; }
        char name[64];
        snprintf(name, sizeof(name), "/layer%02d_input.bin", lid);
        // The buffer holds one row per batch entry; take the last token's row.
        const float * row = lin + (size_t)(seq - 1) * n_embd;
        if (!write_floats(outdir + name, row, (size_t) n_embd)) failures++;
    }

    // Manifest so the comparison harness knows what it is looking at.
    FILE * mf = fopen((outdir + "/manifest.json").c_str(), "w");
    if (mf) {
        fprintf(mf, "{\n");
        fprintf(mf, "  \"model\": \"%s\",\n", model_path);
        fprintf(mf, "  \"n_vocab\": %d,\n", n_vocab);
        fprintf(mf, "  \"n_embd\": %d,\n", n_embd);
        fprintf(mf, "  \"n_layers\": %d,\n", n_layers);
        fprintf(mf, "  \"n_tokens\": %d,\n", seq);
        fprintf(mf, "  \"tokens\": [");
        for (size_t i = 0; i < tokens.size(); ++i) {
            fprintf(mf, "%s%d", i ? "," : "", tokens[i]);
        }
        fprintf(mf, "],\n");
        fprintf(mf, "  \"tensors\": [\"logits\", \"final_hidden\"");
        for (int lid : want_layers) fprintf(mf, ", \"layer%02d_input\"", lid);
        fprintf(mf, "],\n");
        fprintf(mf, "  \"layout\": \"float32 little-endian; layerNN_input is the last token row only\"\n");
        fprintf(mf, "}\n");
        fclose(mf);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "done, failures=%d\n", failures);
    return failures ? 1 : 0;
}
