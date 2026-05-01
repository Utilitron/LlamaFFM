package ffm.llama.session;

import ffm.llama.context.LlamaContext;
import ffm.llama.context.state.CachedContextState;
import ffm.llama.service.LlmService;

import java.util.Optional;

/**
 * Abstraction for KV cache snapshot/restore.
 */
public interface StateSerializer {
    Optional<CachedContextState> snapshot(LlmService.ModelInstance instance);
    boolean restoreContext(LlamaContext context, CachedContextState state);
}
