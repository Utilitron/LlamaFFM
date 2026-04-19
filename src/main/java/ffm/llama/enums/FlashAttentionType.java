package ffm.llama.enums;

public enum FlashAttentionType {
    AUTO(-1),
    DISABLED(0),
    ENABLED(1);

    private final int value;
    FlashAttentionType(int value) { this.value = value; }
    public int getValue() { return value; }
}