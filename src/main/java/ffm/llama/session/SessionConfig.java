package ffm.llama.session;

/**
 * Configuration for a generation session.
 */
public record SessionConfig(
        boolean verbose,
        boolean trackMetrics
) {
    
    public static SessionConfig defaults() {
        return new SessionConfig(false, true);
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private boolean verbose = false;
        private boolean trackMetrics = true;
        
        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }
        
        public Builder trackMetrics(boolean trackMetrics) {
            this.trackMetrics = trackMetrics;
            return this;
        }
        
        public SessionConfig build() {
            return new SessionConfig(verbose, trackMetrics);
        }
    }
}