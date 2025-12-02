param(
	[string]$Notebook = "CarPriceProject.ipynb"
)

$ErrorActionPreference = 'Stop'
. .\.venv\Scripts\Activate.ps1
jupyter notebook $Notebook

