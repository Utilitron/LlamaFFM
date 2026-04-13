package ffm.llama.utils;

import java.util.*;
import java.util.function.Predicate;

public class TemplateDetector {

    public enum LlmChatTemplate {
        UNKNOWN("unknown"),
        CHATML("chatml"),
        LLAMA_2("llama2"),
        LLAMA_2_SYS("llama2-sys"),
        LLAMA_2_SYS_BOS("llama2-sys-bos"),
        LLAMA_2_SYS_STRIP("llama2-sys-strip"),
        MISTRAL_V1("mistral-v1"),
        MISTRAL_V3("mistral-v3"),
        MISTRAL_V3_TEKKEN("mistral-v3-tekken"),
        MISTRAL_V7("mistral-v7"),
        MISTRAL_V7_TEKKEN("mistral-v7-tekken"),
        PHI_3("phi3"),
        PHI_4("phi4"),
        FALCON_3("falcon3"),
        ZEPHYR("zephyr"),
        MONARCH("monarch"),
        GEMMA("gemma"),
        ORION("orion"),
        OPENCHAT("openchat"),
        VICUNA("vicuna"),
        VICUNA_ORCA("vicuna-orca"),
        DEEPSEEK("deepseek"),
        DEEPSEEK_2("deepseek2"),
        DEEPSEEK_3("deepseek3"),
        DEEPSEEK_OCR("deepseek-ocr"),
        COMMAND_R("command-r"),
        LLAMA_3("llama3"),
        CHATGLM_3("chatglm3"),
        CHATGLM_4("chatglm4"),
        GLMEDGE("glmedge"),
        MINICPM("minicpm"),
        EXAONE_3("exaone3"),
        EXAONE_4("exaone4"),
        EXAONE_MOE("exaone-moe"),
        RWKV_WORLD("rwkv-world"),
        GRANITE_3_X("granite"),
        GRANITE_4_0("granite-4.0"),
        GIGACHAT("gigachat"),
        MEGREZ("megrez"),
        YANDEX("yandex"),
        BAILING("bailing"),
        BAILING_THINK("bailing-think"),
        BAILING2("bailing2"),
        LLAMA4("llama4"),
        SMOLVLM("smolvlm"),
        DOTS1("dots1"),
        HUNYUAN_MOE("hunyuan-moe"),
        OPENAI_MOE("gpt-oss"),
        HUNYUAN_DENSE("hunyuan-dense"),
        HUNYUAN_OCR("hunyuan-ocr"),
        KIMI_K2("kimi-k2"),
        SEED_OSS("seed_oss"),
        GROK_2("grok-2"),
        PANGU_EMBED("pangu-embedded"),
        SOLAR_OPEN("solar-open");

        private final String name;
        private static final Map<String, LlmChatTemplate> BY_NAME = new HashMap<>();

        static {
            for (LlmChatTemplate t : values()) {
                BY_NAME.put(t.name, t);
            }
        }

        LlmChatTemplate(String name) {
            this.name = name;
        }

        public String getName() {
            return name;
        }

        public static LlmChatTemplate fromString(String text) {
            return BY_NAME.getOrDefault(text, UNKNOWN);
        }

        @Override
        public String toString() {
            return name;
        }
    }

    public static String getTemplateName(String template) {
        return detectTemplate(template).getName();
    }

    public static LlmChatTemplate detectTemplate(String template) {
        if (template == null || template.isEmpty()) return LlmChatTemplate.UNKNOWN;

        Predicate<String> contains = template::contains;

        // Logic sorted by specificity to avoid false positives
        if (contains.test("<|im_start|>")) {
            if (contains.test("<|im_sep|>")) return LlmChatTemplate.PHI_4;
            if (contains.test("<end_of_utterance>")) return LlmChatTemplate.SMOLVLM;
            return LlmChatTemplate.CHATML;
        }

        if (template.startsWith("mistral") || contains.test("[INST]")) {
            if (contains.test("[SYSTEM_PROMPT]")) return LlmChatTemplate.MISTRAL_V7;
            if (contains.test("' [INST] ' + system_message") || contains.test("[AVAILABLE_TOOLS]")) {
                if (contains.test(" [INST]")) return LlmChatTemplate.MISTRAL_V1;
                if (contains.test("\"[INST]\"")) return LlmChatTemplate.MISTRAL_V3_TEKKEN;
                return LlmChatTemplate.MISTRAL_V3;
            }
            // Llama 2 style fallbacks
            if (contains.test("content.strip()")) return LlmChatTemplate.LLAMA_2_SYS_STRIP;
            if (contains.test("bos_token + '[INST]")) return LlmChatTemplate.LLAMA_2_SYS_BOS;
            if (contains.test("<<SYS>>")) return LlmChatTemplate.LLAMA_2_SYS;
            return LlmChatTemplate.LLAMA_2;
        }

        if (contains.test("<|start_header_id|>")) return LlmChatTemplate.LLAMA_3;
        if (contains.test("<|header_start|>")) return LlmChatTemplate.LLAMA4;
        if (contains.test("<|assistant|>") && contains.test("<|end|>")) return LlmChatTemplate.PHI_3;
        if (contains.test("<start_of_turn>")) return LlmChatTemplate.GEMMA;
        if (contains.test("<|START_OF_TURN_TOKEN|>")) return LlmChatTemplate.COMMAND_R;

        // DeepSeek Check
        if (contains.test("<｜Assistant｜>") || contains.test("<｜User｜>")) {
            return contains.test("<｜Assistant｜>") ? LlmChatTemplate.DEEPSEEK_3 : LlmChatTemplate.DEEPSEEK_2;
        }

        // Granite Check
        if (contains.test("<|start_of_role|>")) {
            return (contains.test("<tool_call>") || contains.test("<tools>"))
                    ? LlmChatTemplate.GRANITE_4_0 : LlmChatTemplate.GRANITE_3_X;
        }

        return LlmChatTemplate.UNKNOWN;
    }
}