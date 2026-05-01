package ffm.llama.batch;

public class DefaultBatchFactory implements BatchFactory {
    @Override
    public LlamaBatch createPrefillBatch(int[] tokens, int startPos, boolean outputLogits) {
        LlamaBatch batch = LlamaBatch.create(tokens.length);
        for (int i = 0; i < tokens.length; i++) {
            boolean emitLogits = outputLogits && (i == tokens.length - 1);
            batch.add(tokens[i], startPos + i, 0, emitLogits);
        }
        return batch;
    }
    
    @Override
    public LlamaBatch createDecodeBatch(int token, int position) {
        return LlamaBatch.forSingleToken(token, position, 0);
    }
}