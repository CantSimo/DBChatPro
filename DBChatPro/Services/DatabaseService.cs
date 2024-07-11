using Microsoft.Data.SqlClient;
using System.Text;
using System.Text.Json;

namespace DBChatPro
{
    public class DatabaseService
    {
        public static List<List<string>> GetDataTable(AIConnection conn, string sqlQuery)
        {
            SqlConnectionStringBuilder builder = new SqlConnectionStringBuilder();

            var rows = new List<List<string>>();
            using (SqlConnection connection = new SqlConnection(conn.ConnectionString))
            {
                connection.Open();

                using (SqlCommand command = new SqlCommand(sqlQuery, connection))
                {
                    using (SqlDataReader reader = command.ExecuteReader())
                    {
                        int count = 0;
                        bool headersAdded = false;
                        while (reader.Read())
                        {
                            var cols = new List<string>();
                            var headerCols = new List<string>();
                            if (!headersAdded)
                            {
                                for (int i = 0; i < reader.FieldCount; i++)
                                {
                                    headerCols.Add(reader.GetName(i).ToString());
                                }
                                headersAdded = true;
                                rows.Add(headerCols);
                            }

                            for (int i = 0; i <= reader.FieldCount - 1; i++)
                            {
                                try
                                {
                                    cols.Add(reader.GetValue(i).ToString());
                                }
                                catch
                                {
                                    cols.Add("DataTypeConversionError");
                                }
                            }
                            rows.Add(cols);
                        }
                    }
                }
            }

            return rows;
        }

        public static string GetDataBlob(AIConnection conn, string sqlQuery)
        {
            StringBuilder results = new StringBuilder();
            SqlConnectionStringBuilder builder = new SqlConnectionStringBuilder();

            var rows = new List<List<string>>();
            using (SqlConnection connection = new SqlConnection(conn.ConnectionString))
            {
                connection.Open();

                using (SqlCommand command = new SqlCommand(sqlQuery, connection))
                {
                    using (SqlDataReader reader = command.ExecuteReader())
                    {
                        int count = 0;
                        bool headersAdded = false;
                        while (reader.Read())
                        {
                            var cols = new List<string>();
                            var headerCols = new List<string>();
                            if (!headersAdded)
                            {
                                for (int i = 0; i < reader.FieldCount; i++)
                                {
                                    results.Append(reader.GetName(i).ToString()).Append(" | ");
                                }
                                headersAdded = true;
                                results.AppendLine();
                            }

                            for (int i = 0; i <= reader.FieldCount - 1; i++)
                            {
                                try
                                {
                                    results.Append(reader.GetValue(i).ToString()).Append(" | ");
                                }
                                catch
                                {
                                    results.Append("DataTypeConversionError").Append(" | ");
                                }
                            }
                            results.AppendLine();
                        }
                    }
                }
            }

            return results.ToString();
        }

        public static List<TableSchema> GetSchema()
        {
            var schema = File.ReadAllText("Schema.txt");

            return JsonSerializer.Deserialize<List<TableSchema>>(schema);
        }

        public static List<AIConnection> GetAIConnections()
        {
            try
            {
                var schema = File.ReadAllText("AIConnections.txt");

                return JsonSerializer.Deserialize<List<AIConnection>>(schema);
            } 
            catch(Exception e)
            {
                return new List<AIConnection>();
            }
        }
    }
}
