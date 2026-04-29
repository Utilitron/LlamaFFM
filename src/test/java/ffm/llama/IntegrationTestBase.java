package ffm.llama;

import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Path;

/**
 * Base class for integration tests that require the native llama library.
 * <p>
 * Automatically skipped if the library cannot be loaded – exactly the same
 * logic used by {@link ffm.llama.binding.LlamaBindings}.
 * <p>
 * Usage: {@code class MyIT extends LlamaIntegrationTest}
 */
@EnabledIf("nativeLibraryAvailable")
public abstract class IntegrationTestBase {
    
    /**
     * Tries to load the native library exactly as LlamaBindings does,
     * using LLAMA_LIB_PATH if set, otherwise standard lookups.
     *
     * @return true if the library loaded successfully
     */
    static boolean nativeLibraryAvailable() {
        try {
            String llamaPath = System.getenv("LLAMA_LIB_PATH");
            if (llamaPath != null) {
                System.load(llamaPath);
            } else {
                System.loadLibrary("llama");
            }
            return true;
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
    }
    
    protected static Path resourceToPath(String resourceName) {
        var url = IntegrationTestBase.class.getClassLoader().getResource(resourceName);
        if (url == null) {
            throw new AssertionError("Test resource not found: " + resourceName);
        }
        return Path.of(url.getPath());
    }
}
