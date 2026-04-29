package ffm.llama.enums;

/**
 * KV Cache Types
 */
public enum KVCacheType {
    F16("f16", 1, 16, 1.0),           // Standard FP16 (baseline)
    Q8_0("q8_0", 8, 8, 2.0),          // Standard 8-bit integer
    Q4_0("q4_0", 2, 4, 4.0),          // Standard 4-bit
    TQ3_0("tq3_0", 42, 3.25, 4.9),    // TurboQuant Stage 1 (~3.25 bits)
    TBQX3("tbqx3", 43, 3.5, 4.7),      // Polar Derotate + Residual
    TBQP3("tbqp3", 44, 3.0, 5.2);      // TurboQuant + QJL (optimal)
    
    private final String llamaCppName;
    private final int nativeId;
    private final double bitsPerWeight;
    private final double compressionRatio;
    
    KVCacheType(String llamaCppName, int nativeId, double bitsPerWeight, double compressionRatio) {
        this.llamaCppName = llamaCppName;
        this.nativeId = nativeId;
        this.bitsPerWeight = bitsPerWeight;
        this.compressionRatio = compressionRatio;
    }
    
    public String getLlamaCppName() {return llamaCppName;}
    
    public int getNativeId() {return nativeId;}
    
    public double getBitsPerWeight() {return bitsPerWeight;}
    
    public double getCompressionRatio() {return compressionRatio;}
    
    /**
     * Estimate VRAM savings for a given context length
     * Formula: contextLength × (16 / bitsPerWeight) × headDim × numHeads × numLayers × 2
     */
    public long estimateVramSavingsMB(int contextLength, int modelSizeB) {
        // Rough approximation for typical architectures
        double baseVramGB = switch (modelSizeB) {
            case 7, 8 -> 4.6;   // 7B/8B models
            case 13, 14 -> 7.2; // 13B/14B models
            case 27, 30 -> 12.5;
            case 70 -> 35.0;
            default -> modelSizeB * 0.5;
        };
        
        double savingsFactor = (1.0 - (bitsPerWeight / 16.0));
        return (long) (baseVramGB * savingsFactor * 1024 * (contextLength / 32768.0));
    }
}
