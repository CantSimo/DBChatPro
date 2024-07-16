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

    public class ResponseData
    {
        public string Response { get; set; }
    }

}
