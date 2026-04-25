package ffm.llama.utils;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static ffm.llama.utils.TemplateDetector.LlmChatTemplate;
import static org.junit.jupiter.api.Assertions.*;

class TemplateDetectorTest {

    @Nested
    @DisplayName("detectTemplate – known template markers")
    class KnownTemplates {

        @Test
        void shouldDetectChatML() {
            assertEquals(LlmChatTemplate.CHATML,
                    TemplateDetector.detectTemplate("<|im_start|>system\nYou are helpful.<|im_end|>"));
        }

        @Test
        void shouldDetectPhi4() {
            // PHI_4 must contain both <|im_start|> and <|im_sep|>
            assertEquals(LlmChatTemplate.PHI_4,
                    TemplateDetector.detectTemplate("<|im_start|>user<|im_sep|>Hello<|im_end|>"));
        }

        @Test
        void shouldDetectSmolVlm() {
            assertEquals(LlmChatTemplate.SMOLVLM,
                    TemplateDetector.detectTemplate("<|im_start|>assistant<end_of_utterance>Hi"));
        }

        @Test
        void shouldDetectMistralV1() {
            assertEquals(TemplateDetector.LlmChatTemplate.MISTRAL_V1,
                    TemplateDetector.detectTemplate(
                            " [INST] [AVAILABLE_TOOLS]{\"tools\":[]}[/AVAILABLE_TOOLS] Hello [/INST]"));
        }

        @Test
        void shouldDetectMistralV7() {
            assertEquals(LlmChatTemplate.MISTRAL_V7,
                    TemplateDetector.detectTemplate("[INST] [SYSTEM_PROMPT] You are helpful.[/INST]"));
        }

        @Test
        void shouldDetectMistralV3Tekken() {
            assertEquals(TemplateDetector.LlmChatTemplate.MISTRAL_V3_TEKKEN,
                    TemplateDetector.detectTemplate(
                            "[AVAILABLE_TOOLS]{\"tools\":[]}[/AVAILABLE_TOOLS] \"[INST]\" Hello \"[/INST]\""));
        }

        @Test
        void shouldDetectLlama2() {
            assertEquals(LlmChatTemplate.LLAMA_2,
                    TemplateDetector.detectTemplate("[INST] Hi [/INST]"));
        }

        @Test
        void shouldDetectLlama2Sys() {
            assertEquals(LlmChatTemplate.LLAMA_2_SYS,
                    TemplateDetector.detectTemplate("[INST] <<SYS>>\nYou are helpful.\n<</SYS>>\n\nHi"));
        }

        @Test
        void shouldDetectLlama3() {
            assertEquals(LlmChatTemplate.LLAMA_3,
                    TemplateDetector.detectTemplate("<|start_header_id|>user<|end_header_id|>Hello"));
        }

        @Test
        void shouldDetectLlama4() {
            assertEquals(LlmChatTemplate.LLAMA4,
                    TemplateDetector.detectTemplate("<|header_start|>user<|header_end|>Hello"));
        }

        @Test
        void shouldDetectPhi3() {
            assertEquals(LlmChatTemplate.PHI_3,
                    TemplateDetector.detectTemplate("<|assistant|>Hello<|end|>"));
        }

        @Test
        void shouldDetectGemma() {
            assertEquals(LlmChatTemplate.GEMMA,
                    TemplateDetector.detectTemplate("<start_of_turn>user\nHi<end_of_turn>"));
        }

        @Test
        void shouldDetectCommandR() {
            assertEquals(LlmChatTemplate.COMMAND_R,
                    TemplateDetector.detectTemplate("<|START_OF_TURN_TOKEN|>User: hello"));
        }

        @Test
        void shouldDetectDeepSeek3() {
            assertEquals(LlmChatTemplate.DEEPSEEK_3,
                    TemplateDetector.detectTemplate("<｜Assistant｜>Hello</s>"));
        }

        @Test
        void shouldDetectDeepSeek2() {
            assertEquals(LlmChatTemplate.DEEPSEEK_2,
                    TemplateDetector.detectTemplate("<｜User｜>Hi"));
        }

        @Test
        void shouldDetectGranite4() {
            assertEquals(LlmChatTemplate.GRANITE_4_0,
                    TemplateDetector.detectTemplate("<|start_of_role|>user<|end_of_role|><tool_call>"));
        }

        @Test
        void shouldDetectGranite3() {
            assertEquals(LlmChatTemplate.GRANITE_3_X,
                    TemplateDetector.detectTemplate("<|start_of_role|>user<|end_of_role|>Hello"));
        }
    }

    @Nested
    @DisplayName("detectTemplate – edge cases")
    class EdgeCases {

        @Test
        void shouldReturnUnknownForNull() {
            assertEquals(LlmChatTemplate.UNKNOWN, TemplateDetector.detectTemplate(null));
        }

        @Test
        void shouldReturnUnknownForEmptyString() {
            assertEquals(LlmChatTemplate.UNKNOWN, TemplateDetector.detectTemplate(""));
        }

        @Test
        void shouldReturnUnknownForUnrecognizedContent() {
            assertEquals(LlmChatTemplate.UNKNOWN,
                    TemplateDetector.detectTemplate("This is some random template that doesn't match anything."));
        }

        @Test
        void shouldNotMisclassifyPartialMarkers() {
            // A template that accidentally contains "<|im_start|" but not the full token should still be CHATML
            assertEquals(LlmChatTemplate.CHATML,
                    TemplateDetector.detectTemplate("<|im_start|>something"));
            // but without the token it's unknown
            assertEquals(LlmChatTemplate.UNKNOWN,
                    TemplateDetector.detectTemplate("<|start_header_id|"));
        }
    }

    @Nested
    @DisplayName("getTemplateName")
    class GetTemplateName {

        @Test
        void shouldReturnNameForKnownTemplate() {
            assertEquals("chatml", TemplateDetector.getTemplateName("<|im_start|>system\n"));
        }

        @Test
        void shouldReturnUnknownForNull() {
            assertEquals("unknown", TemplateDetector.getTemplateName(null));
        }
    }

    @Nested
    @DisplayName("LlmChatTemplate enum")
    class LlmChatTemplateEnum {

        @Test
        void fromStringShouldReturnCorrectEnum() {
            assertEquals(LlmChatTemplate.CHATML, LlmChatTemplate.fromString("chatml"));
            assertEquals(LlmChatTemplate.MISTRAL_V1, LlmChatTemplate.fromString("mistral-v1"));
        }

        @Test
        void fromStringShouldReturnUnknownForInvalidName() {
            assertEquals(LlmChatTemplate.UNKNOWN, LlmChatTemplate.fromString("nonexistent"));
        }

        @Test
        void fromStringShouldBeCaseSensitive() {
            // The map is built with exact names, so case matters.
            assertEquals(LlmChatTemplate.UNKNOWN, LlmChatTemplate.fromString("CHATML"));
        }
    }
}