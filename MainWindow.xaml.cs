using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace EpidemicSimulator
{
    public partial class MainWindow : Window
    {
        private Process pythonProcess;

        public MainWindow()
        {
            InitializeComponent();
        }

        private async void BtnRunSimulation_Click(object sender, RoutedEventArgs e)
        {
            btnRunSimulation.IsEnabled = false;
            txtOutput.Document.Blocks.Clear();
            AppendOutput("Starting Python simulation...\n");

            string pythonPath = "python";
            string scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "simulation_interface.py");

            if (!File.Exists(scriptPath))
            {
                AppendOutput($"[ERROR] Python script not found at {scriptPath}.");
                AppendOutput("Please ensure 'simulation_interface.py' is in the same directory as this executable.");
                btnRunSimulation.IsEnabled = true;
                return;
            }

            var arguments = new StringBuilder();
            arguments.Append($"{scriptPath} ");
            arguments.Append($"--grid_size {txtGridSize.Text} ");
            arguments.Append($"--area_population {txtAreaPopulation.Text} ");
            arguments.Append($"--steps {txtSteps.Text} ");
            arguments.Append($"--generations {txtGenerations.Text} ");
            if (chkAnimate.IsChecked == true)
            {
                arguments.Append("--animate");
            }

            try
            {
                pythonProcess = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = pythonPath,
                        Arguments = arguments.ToString(),
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };

                pythonProcess.OutputDataReceived += (s, ev) =>
                {
                    if (!string.IsNullOrEmpty(ev.Data))
                    {
                        Dispatcher.Invoke(() => AppendOutput(ev.Data));
                    }
                };

                pythonProcess.ErrorDataReceived += (s, ev) =>
                {
                    if (!string.IsNullOrEmpty(ev.Data))
                    {
                        Dispatcher.Invoke(() => AppendOutput("[ERROR] " + ev.Data));
                    }
                };

                pythonProcess.Start();
                pythonProcess.BeginOutputReadLine();
                pythonProcess.BeginErrorReadLine();

                await Task.Run(() => pythonProcess.WaitForExit());

                Dispatcher.Invoke(() =>
                {
                    AppendOutput("\n--- Simulation Complete ---");
                    AppendOutput($"Exit Code: {pythonProcess.ExitCode}");
                });
            }
            catch (Exception ex)
            {
                Dispatcher.Invoke(() =>
                {
                    AppendOutput($"An error occurred: {ex.Message}");
                    AppendOutput("Please ensure Python is installed and in your system's PATH.");
                });
            }
            finally
            {
                Dispatcher.Invoke(() => btnRunSimulation.IsEnabled = true);
            }
        }

        private void AppendOutput(string text)
        {
            txtOutput.AppendText(text + Environment.NewLine);
            txtOutput.ScrollToEnd();
        }
    }
}
