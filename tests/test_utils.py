import util

def test_log_and_print(tmp_path):
    log_file_path = tmp_path / "log.txt"
    with open(log_file_path, "w") as log_file:
        util.log_and_print("Test message", log_file)

    # Check if the message was printed to the console and written to the log file
    with open(log_file_path, "r") as log_file:
        logged_message = log_file.read().strip()
        assert logged_message == "Test message"

def test_plot_metrics(tmp_path):
    metric_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    title = "Test Metrics"
    ylabel = "Value"
    output_path = tmp_path / "metrics_plot.png"

    util.plot_metrics(metric_values, title, ylabel, output_path)

    # Check if the plot file was created
    assert output_path.exists()