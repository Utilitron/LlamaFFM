package ffm.llama.binding;

/**
 * Probe‑safe loader for the native <code>llama</code> shared library.
 * <p>
 * The library is loaded exactly once, on the first successful call to
 * {@link #load()}.  Before that, you can safely check
 * {@link #isLoaded()} / {@link #isAttempted()} without triggering a
 * JVM‑crashing {@link UnsatisfiedLinkError} (the error will still be
 * thrown, but only inside {@code load()} itself and after the class
 * has been fully initialised).
 * <p>Environment variables:
 * <ul>
 *   <li>{@value #LLAMA_LIB_PATH_ENV} – absolute path to the native
 *       library file (e.g., <code>/opt/llama/libllama.so</code>).
 *       If set, {@code System.load(path)} is used; otherwise
 *       {@code System.loadLibrary("llama")} relies on the standard
 *       <code>java.library.path</code>.</li>
 * </ul>
 *
 * <p>Thread‑safety: all public methods are either atomic read‑only or
 * {@code synchronized} so that initialisation happens exactly once.
 */
public final class LlamaLoader {
    private static volatile boolean loaded = false;
    private static volatile boolean attempted = false;
    private static UnsatisfiedLinkError loadException = null;
    
    private LlamaLoader() {}
    
    /**
     * Loads the native library exactly once.
     *
     * @throws UnsatisfiedLinkError  if the library cannot be found
     * @throws IllegalStateException if a previous attempt already failed
     */
    public static synchronized void load() throws UnsatisfiedLinkError {
        if (attempted) {
            if (!loaded) throw new IllegalStateException("llama library failed to load previously", loadException);
            return;
        }
        attempted = true;
        try {
            String path = System.getenv("LLAMA_LIB_PATH");
            if (path != null) {
                System.load(path);
            } else {
                System.loadLibrary("llama");
            }
            loaded = true;
        } catch (UnsatisfiedLinkError e) {
            loadException = e;
            throw e;
        }
    }
    
    public static boolean isLoaded() { return loaded; }
    public static boolean isAttempted() { return attempted; }
}