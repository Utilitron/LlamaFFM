package ffm.llama.utils;

import ffm.llama.utils.TemplateDetector.LlmChatTemplate;

/**
 * Handles the specific 'dialects' of tool injection for various LLM architectures.
 * This ensures the model weights recognize tool definitions as functional constraints
 * rather than just plain text.
 */
public class LlmToolGrammar {

    public static String injectTools(LlmChatTemplate template, String toolsJson) {
        if (toolsJson == null || toolsJson.isEmpty()) return "";

        return switch (template) {

            // --- THE MISTRAL BLOCK ---
            // Ministral/Mistral is extremely picky: No spaces between the tags and JSON.
            case MISTRAL_V1, MISTRAL_V3, MISTRAL_V3_TEKKEN, MISTRAL_V7, MISTRAL_V7_TEKKEN ->
                    "[AVAILABLE_TOOLS]" + toolsJson.trim() + "[/AVAILABLE_TOOLS]";

            // --- THE LLAMA & CHATML BLOCK ---
            case LLAMA_3, LLAMA4, CHATML, ORION, OPENCHAT, SOLAR_OPEN ->
                    "\n\n# Tools\nYou have access to these functions. To call one, respond with JSON.\n" + toolsJson;

            // --- THE PHI BLOCK ---
            case PHI_3, PHI_4 ->
                    "\n\n## Tools\nAvailable for use when necessary:\n" + toolsJson;

            // --- THE DEEPSEEK BLOCK ---
            case DEEPSEEK, DEEPSEEK_2, DEEPSEEK_3 ->
                    "\nTools available:\n" + toolsJson + "\n\n";

            // --- THE COMMAND-R BLOCK ---
            case COMMAND_R ->
                    "\n<TOOL_DEFINITION>\n" + toolsJson + "\n</TOOL_DEFINITION>";

            // --- THE IBM GRANITE BLOCK ---
            case GRANITE_3_X, GRANITE_4_0 ->
                    "\n<tools>\n" + toolsJson + "\n</tools>";

            // --- THE CHATGLM & BAILING BLOCK ---
            case CHATGLM_3, CHATGLM_4, GLMEDGE, BAILING, BAILING2, BAILING_THINK, HUNYUAN_MOE, HUNYUAN_DENSE ->
                    "\n[TOOL_DEFINITION]\n" + toolsJson + "\n[/TOOL_DEFINITION]";

            // --- THE OPENAI_MOE / GPT-OSS BLOCK ---
            case OPENAI_MOE ->
                    "## Functions\n\n" + toolsJson + "\n\n";

            default -> "\n\nAvailable tools:\n" + toolsJson + "\n";
        };
    }
}