package ffm.llama.session;

import ffm.llama.config.ModelConfig;
import ffm.llama.context.LlamaContext;
import ffm.llama.context.state.CachedContextState;
import ffm.llama.model.LlamaModel;
import ffm.llama.service.LlmService;

import java.util.Optional;

/**
 * Abstraction for KV cache snapshot/restore.
 */
public interface StateSerializer {
    Optional<CachedContextState> snapshot(LlamaModel model, LlamaContext context, ModelConfig config);
    boolean restoreContext(LlamaContext context, CachedContextState state);
}
