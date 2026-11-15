```markdown
# A Pointwise \(\phi\)-Variance Wall Function with Calibrated RANS Inputs for Turbulent Channel Flow

This repository contains the code, data, and supplementary material for the paper:

> **A Pointwise \(\phi\)-Variance Wall Function with Calibrated RANS Inputs for Turbulent Channel Flow**  
> Mahmoud F. Abdel-Sattar  
> *Journal of Turbulence* (submitted, 2026)

## Overview

Accurate near-wall modelling of second-moment turbulence statistics remains a challenge in Reynolds-averaged Navier–Stokes (RANS) computations. This work assesses the pointwise performance of a \(\phi\)-variance wall function when supplied with inputs from the standard \(k\)-\(\varepsilon\) model.

The wall function is based on the inverse hyperbolic sine transformation
\[
\phi = \operatorname{arcsinh}\!\left(\frac{U^+}{U_{\mathrm{ref}}^+}\right),
\]
and the exact quadratic identity
\[
\langle\phi^2\rangle = \phi_0^2 + \langle\phi'^2\rangle.
\]
Under the linearized approximation for small fluctuations, this leads to
\[
\langle\phi'^2\rangle_{\mathrm{WF}}
=
\frac{\langle u'^2\rangle_{\mathrm{model}}}{1+U^{+2}},
\qquad
\langle u'^2\rangle_{\mathrm{model}}
=
\frac{2}{3}k\left[1 + C_u\,\frac{\nu_t}{k}\left|\frac{dU}{dy}\right|\right],
\]
where \(C_u\) is an empirical anisotropy correction calibrated for each Reynolds number.

The wall function is evaluated at the first interior RANS node for \(Re_\tau = 180, 395, 590\), using DNS data of Moser, Kim & Mansour (1999) as the reference. With a calibrated \(C_u\), the relative errors are \(2.2\%\), \(9.5\%\), and \(12.6\%\), respectively. A constant \(C_u = 1.5\) yields errors of \(34.7\%\), \(13.0\%\), and \(41.5\%\), confirming that a single constant is not universally applicable. A preliminary empirical relation \(C_u(Re_\tau) \approx -0.147 + 556.357/Re_\tau\) is proposed as a guideline.

## Repository Structure

```text
phi-variance-wall-function-rans/
├── codes/
│   ├── main_analysis.py          # Main script: loads DNS, solves RANS,
│   │                             # evaluates wall function, calibrates C_u,
│   │                             # produces figures and results.
│   └── colab_upload_data.py      # Optional helper for Google Colab.
├── data/
│   ├── README.md                 # Source, license, and usage of DNS data.
│   └── chandata.tar.gz           # Compressed DNS data (Moser et al. 1999).
├── figures/                      # Output figures (created on run).
│   ├── pointwise_error_vs_Cu.png
│   └── optimal_Cu_vs_Re.png
├── results/                      # Output results (created on run).
│   └── pointwise_results.json
├── README.md                     # This file.
├── requirements.txt              # Python dependencies.
├── LICENSE                       # Code license (MIT).
└── .gitignore                    # Files to be ignored by Git.
```

Requirements

· Python 3.8 or higher
· numpy
· scipy
· matplotlib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Data

The DNS data used in this study are the publicly available channel-flow data of

R. D. Moser, J. Kim, and N. N. Mansour,
Direct numerical simulation of turbulent channel flow up to Re_\tau = 590,
Physics of Fluids, vol. 11, no. 4, pp. 943–945, 1999.

The data are provided in data/chandata.tar.gz. They are used for research purposes only. Please refer to data/README.md for details.

If you prefer to download the original dataset from the official source, see:

· Moser, Kim & Mansour (1999) channel flow DNS data

The expected folder structure after extraction is:

```text
chan180/profiles/
chan395/profiles/
chan590/profiles/
```

Each profiles/ folder should contain the files:

· chanXXX.means
· chanXXX.reystress

Usage

Local run

1. Ensure the DNS data folders (chan180, chan395, chan590) are present in the working directory. If you have data/chandata.tar.gz, extract it:
   ```bash
   tar -xzf data/chandata.tar.gz
   ```
2. Run the main analysis:
   ```bash
   python codes/main_analysis.py
   ```

Google Colab

If you are running on Google Colab and the data folders are not present, you will be prompted to upload chandata.tar.gz. Alternatively, you can run the helper script first:

```python
from google.colab import files
uploaded = files.upload()
# Select chandata.tar.gz
```

Or simply run codes/colab_upload_data.py, which handles the upload and extraction.

Outputs

After a successful run, the following files are produced:

· figures/pointwise_error_vs_Cu.png — relative error in \phi-variance vs C_u for each Re_\tau.
· figures/optimal_Cu_vs_Re.png — optimal C_u vs Re_\tau with empirical fit.
· results/pointwise_results.json — numerical results for all cases.

The console output also reports:

· The first interior RANS node location y_1^+.
· DNS and RANS values at that node.
· Optimal C_u and relative error for each Re_\tau.
· Errors with a constant C_u = 1.5.
· The empirical fit C_u(Re_\tau).

Reproducibility

The results presented in the paper can be reproduced exactly with the provided code and data. The two figures and the JSON results file are regenerated deterministically. The RANS solver is a simple finite-volume channel solver with wall functions; no external CFD libraries are required.

If you use this code in your own research, please cite the paper (see below).

Citation

If you use this code or data in your work, please cite:

```bibtex
@article{AbdelSattar2025,
  author  = {Abdel-Sattar, Mahmoud F.},
  title   = {A Pointwise $\phi$-Variance Wall Function with Calibrated RANS Inputs for Turbulent Channel Flow},
  journal = {Journal of Turbulence},
  year    = {2025},
  note    = {Submitted}
}
```

License

The code in this repository is released under the MIT License. See LICENSE for details.

The DNS data are subject to the terms of the original source (Moser, Kim & Mansour, 1999). They are included here for convenience and research use only. Please acknowledge the original authors when using the data.

Contact

Mahmoud F. Abdel-Sattar
Department of Astronomy and Meteorology
Faculty of Science, Al-Azhar University
Cairo, Egypt

· ORCID: 0009-0003-0224-7007
· Email: m.f.abdel-sattar@azhar.edu.eg

For questions, suggestions, or bug reports, please open an issue on GitHub or contact the author directly.

Acknowledgements

The author gratefully acknowledges the developers of NumPy, SciPy, and Matplotlib, and the authors of the DNS database used in this study.

```
