namespace DBChatPro
{
    public static class HistoryService
    {
        private static List<HistoryItem> Queries = new();
        private static List<HistoryItem> Favorites = new();

        public static void SaveFavorite(string query, string connectionName)
        {
            Favorites.Add(new HistoryItem()
            {
                Id = new Random().Next(0, 10000),
                Query = query,
                Name = query,
                ConnectionName = connectionName
            });
        }

        public static void SaveQuery(string query, string connectionName)
        {
            Queries.Add(new HistoryItem()
            {
                Id = new Random().Next(0, 10000),
                Query = query,
                Name = query,
                ConnectionName = connectionName
            });
        }

        public static List<HistoryItem> GetQueries(string connectionName)
        {
            return Queries.Where(x => x.ConnectionName == connectionName).ToList();
        }

        public static List<HistoryItem> GetFavorites(string connectionName)
        {
            return Favorites.Where(x => x.ConnectionName == connectionName).ToList();
        }
    }

    public class HistoryItem
    {
        public int Id { get; set; }
        public string Query { get; set; }
        public string Name { get; set; }
        public string ConnectionName { get; set; }
    }
}
