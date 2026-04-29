package ffm.llama.session;

import ffm.llama.model.LlamaContext;
import ffm.llama.model.state.CachedContextState;
import ffm.llama.service.LlmService;

import java.util.Optional;

/**
 * Abstraction for KV cache snapshot/restore.
 */
public interface StateSerializer {
    Optional<CachedContextState> snapshot(LlmService.ModelInstance instance);
    boolean restoreContext(LlamaContext context, CachedContextState state);
}
