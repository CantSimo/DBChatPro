namespace DBChatPro
{
    public class AIConnection
    {
        public string ConnectionString { get; set;}
        public string Name { get; set; }
        public List<TableSchema> SchemaStructured { get; set; }
        public List<string> SchemaRaw { get; set; }
    }
}
