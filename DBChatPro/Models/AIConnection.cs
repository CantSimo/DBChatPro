using System.Text.Json.Serialization;

namespace DBChatPro
{
    public class AIConnection
    {
        public string ConnectionString { get; set;}
        public string Name { get; set; }
        public List<TableSchema> SchemaStructured { get; set; }
        public List<string> SchemaRaw { get; set; }
    }

    public class RequestData
    {
        public string Query { get; set; } = "";
        public string Model { get; set; } = "";
        public double Temperature { get; set; }
        public List<string> Chat_History { get; set; } = new List<string>();
    }
    public class OpenAIApiResponse
    {
        [JsonPropertyName("response")]
        public ResponseData Response { get; set; }
    }

    public class ResponseData
    {
        [JsonPropertyName("response")]
        public string Query { get; set; }

        [JsonPropertyName("openai_fee")]
        public OpenAIFee OpenAIFee { get; set; }
    }

    public class OpenAIFee
    {
        [JsonPropertyName("completion_tokens")]
        public int CompletionTokens { get; set; }

        [JsonPropertyName("prompt_tokens")]
        public int PromptTokens { get; set; }

        [JsonPropertyName("total_cost")]
        public double TotalCost { get; set; }

        public override string ToString()
        {
            return $"completion_tokens:[{CompletionTokens.ToString()}]\nprompt_tokens:[{PromptTokens.ToString()}]\ntotal_cost:[{TotalCost.ToString()}$]";
        }
    }
}
