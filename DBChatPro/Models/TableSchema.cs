namespace DBChatPro
{
    public class TableSchema()
    {
        public string TableName { get; set; }
        public List<ColumnSchema> Columns { get; set; }
    }

    public class ColumnSchema
    {
        public string ColumnName { get; set; }
        public string DataType { get; set; }
        public string MaxLength { get; set; }

        public override string ToString()
        {
            return ColumnName + " " + DataType + " " + MaxLength;
        }
    }
}
