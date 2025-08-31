using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

public class EpidemicSimulatorForm : Form
{
    private Label lblGridSize;
    private TextBox txtGridSize;
    private Label lblAreaPopulation;
    private TextBox txtAreaPopulation;
    private Label lblSteps;
    private TextBox txtSteps;
    private Label lblGenerations;
    private TextBox txtGenerations;
    private CheckBox chkAnimate;
    private Button btnRunSimulation;
    private RichTextBox txtOutput;
    private Process pythonProcess;

    public EpidemicSimulatorForm()
    {
        InitializeComponent();
    }

    private void InitializeComponent()
    {
        // Form properties
        this.Text = "Epidemic Simulation Control Panel";
        this.Size = new Size(800, 600);
        this.MinimumSize = new Size(600, 400);
        this.Font = new Font("Segoe UI", 9);
        this.Padding = new Padding(10);

        // Controls initialization
        lblGridSize = new Label { Text = "Grid Size:", AutoSize = true, Location = new Point(10, 20) };
        txtGridSize = new TextBox { Text = "2", Location = new Point(120, 15), Width = 150 };

        lblAreaPopulation = new Label { Text = "Area Population:", AutoSize = true, Location = new Point(10, 50) };
        txtAreaPopulation = new TextBox { Text = "100,100,100,100", Location = new Point(120, 45), Width = 150 };

        lblSteps = new Label { Text = "Steps (days):", AutoSize = true, Location = new Point(10, 80) };
        txtSteps = new TextBox { Text = "365", Location = new Point(120, 75), Width = 150 };

        lblGenerations = new Label { Text = "Generations:", AutoSize = true, Location = new Point(10, 110) };
        txtGenerations = new TextBox { Text = "15", Location = new Point(120, 105), Width = 150 };

        chkAnimate = new CheckBox { Text = "Show Animation", AutoSize = true, Location = new Point(120, 140) };

        btnRunSimulation = new Button { Text = "Run Simulation", Location = new Point(10, 170), Size = new Size(260, 30) };

        txtOutput = new RichTextBox
        {
            Location = new Point(10, 210),
            Dock = DockStyle.Fill,
            Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
            ReadOnly = true,
            BackColor = Color.FromArgb(240, 240, 240)
        };

        // Add controls to the form
        this.Controls.Add(lblGridSize);
        this.Controls.Add(txtGridSize);
        this.Controls.Add(lblAreaPopulation);
        this.Controls.Add(txtAreaPopulation);
        this.Controls.Add(lblSteps);
        this.Controls.Add(txtSteps);
        this.Controls.Add(lblGenerations);
        this.Controls.Add(txtGenerations);
        this.Controls.Add(chkAnimate);
        this.Controls.Add(btnRunSimulation);
        this.Controls.Add(txtOutput);

        // Event handler
        btnRunSimulation.Click += new EventHandler(BtnRunSimulation_Click);
    }

    private async void BtnRunSimulation_Click(object sender, EventArgs e)
    {
        btnRunSimulation.Enabled = false;
        txtOutput.Clear();
        AppendOutput("Starting Python simulation...\n");

        string pythonPath = "python";
        string scriptPath = Path.Combine(Directory.GetCurrentDirectory(), "simulation_interface.py");

        if (!File.Exists(scriptPath))
        {
            AppendOutput($"[ERROR] Python script not found at {scriptPath}.");
            AppendOutput("Please ensure 'simulation_interface.py' is in the same directory as this executable.");
            btnRunSimulation.Enabled = true;
            return;
        }

        var arguments = new StringBuilder();
        arguments.Append($"{scriptPath} ");
        arguments.Append($"--grid_size {txtGridSize.Text} ");
        arguments.Append($"--area_population {txtAreaPopulation.Text} ");
        arguments.Append($"--steps {txtSteps.Text} ");
        arguments.Append($"--generations {txtGenerations.Text} ");
        if (chkAnimate.Checked)
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
                    AppendOutput(ev.Data);
                }
            };

            pythonProcess.ErrorDataReceived += (s, ev) =>
            {
                if (!string.IsNullOrEmpty(ev.Data))
                {
                    AppendOutput("[ERROR] " + ev.Data);
                }
            };

            pythonProcess.Start();
            pythonProcess.BeginOutputReadLine();
            pythonProcess.BeginErrorReadLine();

            await Task.Run(() => pythonProcess.WaitForExit());

            AppendOutput("\n--- Simulation Complete ---");
            AppendOutput($"Exit Code: {pythonProcess.ExitCode}");
        }
        catch (Exception ex)
        {
            AppendOutput($"An error occurred: {ex.Message}");
            AppendOutput("Please ensure Python is installed and in your system's PATH.");
        }
        finally
        {
            btnRunSimulation.Enabled = true;
        }
    }

    private void AppendOutput(string text)
    {
        if (txtOutput.InvokeRequired)
        {
            txtOutput.Invoke(new Action<string>(AppendOutput), text);
            return;
        }
        txtOutput.AppendText(text + Environment.NewLine);
        txtOutput.ScrollToCaret();
    }

    public static void Main()
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new EpidemicSimulatorForm());
    }
}
