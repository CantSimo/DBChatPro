namespace DBChatPro.Models
{
    public class FormModel
    {
        public string Prompt { get; set; } = String.Empty;
    }

    public enum MyChartTypes
    {
        None,
        Pie,
        HorizontalBar,
        VerticalBar,
        Radar,
        Doughnut,
        PolarArea
    }
}
