using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace GA_MonteCarlo
{
    public partial class MainWindow : Window
    {
        private Process pythonProcess;

        public MainWindow()
        {
            InitializeComponent();
            LoadReplayFiles();
        }

        private void LoadReplayFiles()
        {
            try
            {
                var files = Directory.GetFiles(AppDomain.CurrentDomain.BaseDirectory, "*.npz");
                lbxReplayFiles.Items.Clear();
                foreach (var file in files)
                {
                    lbxReplayFiles.Items.Add(Path.GetFileName(file));
                }
            }
            catch (Exception ex)
            {
                AppendOutput($"[ERROR] Could not load replay files: {ex.Message}");
            }
        }

        private async void BtnRunSimulation_Click(object sender, RoutedEventArgs e)
        {
            // Disable run button and enable cancel button
            btnRunSimulation.IsEnabled = false;
            btnCancelSimulation.IsEnabled = true;

            txtOutput.Document.Blocks.Clear();
            AppendOutput("Starting Python simulation...\n");

            string pythonPath = "python";
            string scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "MonteCarlo.py");

            if (!File.Exists(scriptPath))
            {
                AppendOutput($"[ERROR] Python script not found at {scriptPath}.");
                AppendOutput("Please ensure 'MonteCarlo.py' is in the same directory as this executable.");
                btnRunSimulation.IsEnabled = true;
                btnCancelSimulation.IsEnabled = false;
                return;
            }

            // Get values from the textboxes
            string areaPopulation = $"{txtAreaA.Text},{txtAreaB.Text},{txtAreaC.Text},{txtAreaD.Text}";
            string steps = txtSteps.Text;
            string generations = txtGenerations.Text;
            string monteCarloRuns = txtMonteCarloRuns.Text;
            string populationSize = txtPopulationSize.Text; // New: Get population size
            string hesitancyRate = txtHesitancyRate.Text;
            string productionRate = txtProductionRate.Text;
            string spoilageRate = txtSpoilageRate.Text;
            string vaccineStockStart = txtVaccineStockStart.Text;
            string vaccinationDelay = txtVaccinationDelay.Text;

            var arguments = new StringBuilder();
            arguments.Append($"-u {scriptPath} ");
            arguments.Append($"--area_population {areaPopulation} ");
            arguments.Append($"--steps {steps} ");
            arguments.Append($"--generations {generations} ");
            arguments.Append($"--monte_carlo_runs {monteCarloRuns} ");
            arguments.Append($"--population_size {populationSize} "); // New: Pass population size
            arguments.Append($"--hesitancy_rate {hesitancyRate} ");
            arguments.Append($"--vaccine_production_rate {productionRate} ");
            arguments.Append($"--vaccine_spoilage_rate {spoilageRate} ");
            arguments.Append($"--vaccine_total {vaccineStockStart} ");
            arguments.Append($"--vaccination_delay_days {vaccinationDelay} ");

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

                // Check if the process exited normally or was killed
                if (pythonProcess.ExitCode == 0)
                {
                    Dispatcher.Invoke(() =>
                    {
                        AppendOutput("\n--- Simulation Complete ---");
                        AppendOutput($"Exit Code: {pythonProcess.ExitCode}");
                    });
                }
                else
                {
                    Dispatcher.Invoke(() =>
                    {
                        AppendOutput("\n--- Simulation Terminated ---");
                        AppendOutput("The process was terminated by the user.");
                    });
                }
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
                // Re-enable run button and disable cancel button
                Dispatcher.Invoke(() =>
                {
                    btnRunSimulation.IsEnabled = true;
                    btnCancelSimulation.IsEnabled = false;
                    LoadReplayFiles(); // Reload files after simulation
                });
            }
        }

        private void BtnCancelSimulation_Click(object sender, RoutedEventArgs e)
        {
            if (pythonProcess != null && !pythonProcess.HasExited)
            {
                try
                {
                    AppendOutput("\nTerminating Python process...");
                    pythonProcess.Kill();
                    AppendOutput("Process terminated.");
                }
                catch (Exception ex)
                {
                    AppendOutput($"[ERROR] Failed to terminate process: {ex.Message}");
                }
            }
        }

        private async void BtnReplay_Click(object sender, RoutedEventArgs e)
        {
            if (lbxReplayFiles.SelectedItem == null)
            {
                AppendOutput("[ERROR] Please select a file to replay.");
                return;
            }

            btnReplay.IsEnabled = false;

            string selectedFile = lbxReplayFiles.SelectedItem.ToString();
            AppendOutput($"Replaying simulation from file: {selectedFile}...\n");

            string pythonPath = "python";
            string scriptPath = (selectedFile == "genetic_algorithm_performance_data.npz")
                ? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "PlotGeneticAlgorithmPerformance.py")
                : Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "MonteCarloReplay.py");

            if (!File.Exists(scriptPath))
            {
                AppendOutput($"[ERROR] Python script not found at {scriptPath}.");
                AppendOutput("Please ensure 'MonteCarloReplay.py' is in the same directory as this executable.");
                btnReplay.IsEnabled = true;
                return;
            }

            var arguments = new StringBuilder();
            arguments.Append($"-u {scriptPath} ");
            arguments.Append($"\"{selectedFile}\"");

            try
            {
                var replayProcess = new Process
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
                    StartInfo.EnvironmentVariables["REPLAY_INTERVAL_MS"] = "100";

                };

                replayProcess.OutputDataReceived += (s, ev) =>
                {
                    if (!string.IsNullOrEmpty(ev.Data))
                    {
                        Dispatcher.Invoke(() => AppendOutput(ev.Data));
                    }
                };

                replayProcess.ErrorDataReceived += (s, ev) =>
                {
                    if (!string.IsNullOrEmpty(ev.Data))
                    {
                        Dispatcher.Invoke(() => AppendOutput("[ERROR] " + ev.Data));
                    }
                };

                replayProcess.Start();
                replayProcess.BeginOutputReadLine();
                replayProcess.BeginErrorReadLine();

                await Task.Run(() => replayProcess.WaitForExit());
                AppendOutput("\n--- Replay Complete ---");
            }
            catch (Exception ex)
            {
                AppendOutput($"[ERROR] An error occurred during replay: {ex.Message}");
            }
            finally
            {
                Dispatcher.Invoke(() =>
                {
                    btnReplay.IsEnabled = true;
                });
            }
        }

        private void AppendOutput(string text)
        {
            txtOutput.AppendText(text + Environment.NewLine);
            txtOutput.ScrollToEnd();
        }
    }
}
