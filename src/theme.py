import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    """Sets the plotting style to high data-ink ratio as per assignment guidelines."""
    plt.ioff() # Turn off interactive mode for scripts
    sns.set_theme(style="white")
    
    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'figure.autolayout': True,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def sequential_palette():
    """Returns a sequential palette (Viridis) which is perceptually uniform."""
    return sns.color_palette("viridis", as_cmap=True)

def categorical_palette():
    """Returns a minimalist categorical palette."""
    return {"Industrial": "#e63946", "Residential": "#457b9d"}
