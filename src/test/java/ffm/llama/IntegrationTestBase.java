package ffm.llama;

import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Path;

/**
 * Base class for integration tests that require the native llama library.
 * <p>
 * Automatically skipped if the library cannot be loaded.
 * Provides access to test models via LLAMA_TEST_MODEL_RESOURCE system property.
 * <p>
 * Usage: {@code class MyIT extends IntegrationTestBase}
 */
@EnabledIf("nativeLibraryAvailable")
public abstract class IntegrationTestBase {
    
    /**
     * Tries to load the native library using llama.lib.path system property
     * (set by Maven failsafe plugin) or LLAMA_LIB_PATH environment variable.
     *
     * @return true if the library loaded successfully
     */
    static boolean nativeLibraryAvailable() {
        try {
            // First check system property (set by Maven)
            String llamaPath = System.getProperty("llama.lib.path");
            if (llamaPath == null) {
                // Fall back to environment variable
                llamaPath = System.getenv("LLAMA_LIB_PATH");
            }
            
            if (llamaPath != null && !llamaPath.isEmpty()) {
                System.load(llamaPath);
            } else {
                System.loadLibrary("llama");
            }
            return true;
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native library not available: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Gets the test model path from system property or environment variable.
     *
     * @return path to test model, or null if not configured
     */
    protected static String getTestModelPath() {
        String modelPath = System.getProperty("llama.test.model");
        if (modelPath == null) {
            modelPath = System.getenv("LLAMA_TEST_MODEL_RESOURCE");
        }
        return modelPath;
    }
    
    /**
     * Converts a resource name to a file system path.
     *
     * @param resourceName the resource name
     * @return the file system path
     * @throws AssertionError if resource not found
     */
    protected static Path resourceToPath(String resourceName) {
        var url = IntegrationTestBase.class.getClassLoader().getResource(resourceName);
        if (url == null) {
            throw new AssertionError("Test resource not found: " + resourceName);
        }
        return Path.of(url.getPath());
    }
    
    /**
     * Checks if a test model is available for integration tests.
     *
     * @return true if test model path is configured
     */
    protected static boolean hasTestModel() {
        String modelPath = getTestModelPath();
        return modelPath != null && !modelPath.isEmpty();
    }
}