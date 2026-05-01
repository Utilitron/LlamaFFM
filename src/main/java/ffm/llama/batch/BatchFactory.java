package ffm.llama.batch;

public interface BatchFactory {
    LlamaBatch createPrefillBatch(int[] tokens, int startPos, boolean outputLogits);
    LlamaBatch createDecodeBatch(int token, int position);
}
