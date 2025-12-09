import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# To plot correlation coefficient for each feature with each others
def plot_numerical_correlation(
    X: pd.DataFrame,
    numerical_features: list,
    save_path: str = None,
) -> None:
    """
    Computes and visualizes the Pearson correlation matrix for numerical features
    using a heatmap.

    Args:
        X (pd.DataFrame): Input dataframe.
        numerical_features (list): List of numerical column names to correlate.
    """
    # Calculate the pairwise correlation of columns, excluding NA/null values
    correlation_matrix = X[numerical_features].corr(method='pearson')

    plt.figure(figsize=(15, 12))
    sns.heatmap(
        correlation_matrix,
        cmap='coolwarm',      # Diverging colormap (Red for pos, Blue for neg)
        vmin=-1, vmax=1,      # Anchor the colormap range
        center=0,             # Center the colormap at 0
        linewidths=.5,
        cbar_kws={'label': 'Correlation Coefficient'}
    )

    plt.title('Numerical Features Correlation Heatmap (Pearson)',
              fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def compare_features_boxplot(
    df: pd.DataFrame, cols: list[str],
    ax=None, colors=['#1f77b4', '#d62728'], save_path=None,
    figsize: tuple[int] = (15, 6)
):
    if len(cols) != 2:
        raise ValueError("cols length must be exactly 2")

    col1, col2 = cols
    corr_val = df[col1].corr(df[col2])

    tmp_fig, tmp_ax = plt.subplots(figsize=figsize)

    melted = df[[col1, col2]].melt(var_name='Feature', value_name='Value')
    palette = {col1: colors[0], col2: colors[1]}

    sns.boxplot(data=melted, x='Feature', y='Value', ax=tmp_ax, palette=palette)
    tmp_ax.set_title(f'Boxplot Comparison: {col1} vs {col2}\nCorr = {corr_val:.4f}',
                    fontsize=15, fontweight='bold')
    tmp_ax.grid(True, axis='y', alpha=0.3)

    if save_path or ax is None:
        tmp_fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if ax is not None:
        ax.clear()

        # replot ke axis subplot
        sns.boxplot(data=melted, x='Feature', y='Value', ax=ax, palette=palette)
        ax.set_title(f'Boxplot Comparison: {col1} vs {col2}\nCorr = {corr_val:.4f}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

    plt.close(tmp_fig)
    return corr_val


def compare_features_distribution(
    df: pd.DataFrame, cols: list[str],
    ax=None, colors=['#1f77b4', '#d62728'],
    save_path: str = None,
    figsize: tuple[int] = (15, 6)
):
    if len(cols) != 2:
        raise ValueError("cols length must be exactly 2")

    col1, col2 = cols
    corr_val = df[col1].corr(df[col2])

    tmp_fig, tmp_ax = plt.subplots(figsize=figsize)

    sns.histplot(
        df[col1], kde=True, stat='density', alpha=0.35,
        ax=tmp_ax, color=colors[0], label=col1
    )
    sns.histplot(
        df[col2], kde=True, stat='density', alpha=0.35,
        ax=tmp_ax, color=colors[1], label=col2
    )

    tmp_ax.set_title(
        f'Distribution Comparison: {col1} vs {col2}\nCorr = {corr_val:.4f}',
        fontsize=12, fontweight='bold'
    )
    tmp_ax.set_xlabel('Value')
    tmp_ax.legend()
    tmp_ax.grid(True, alpha=0.3)

    if save_path:
        tmp_fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if ax is not None:
        ax.clear()

        sns.histplot(
            df[col1], kde=True, stat='density', alpha=0.35,
            ax=ax, color=colors[0], label=col1
        )

        sns.histplot(
            df[col2], kde=True, stat='density', alpha=0.35,
            ax=ax, color=colors[1], label=col2
        )

        ax.set_title(
            f'Distribution Comparison: {col1} vs {col2}\nCorr = {corr_val:.4f}',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.close(tmp_fig)
    return corr_val


# To plot target
def check_transformations(series, target, bins=50):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # --- Original Data ---
    sns.histplot(series, bins=bins, kde=True, ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_title(f'Original\n(Skew: {series.skew():.2f})', fontsize=14)

    # --- Log Transformation ---
    series_log = np.log(series[series > 0])
    sns.histplot(series_log, bins=bins, kde=True, ax=axes[1], color='salmon', edgecolor='black')
    axes[1].set_title(f'Log Transform\n(Skew: {series_log.skew():.2f})', fontsize=14)

    # --- Square Root Transformation ---
    series_sqrt = np.sqrt(series)
    sns.histplot(series_sqrt, bins=bins, kde=True, ax=axes[2], color='lightgreen', edgecolor='black')
    axes[2].set_title(f'Square Root Transform\n(Skew: {series_sqrt.skew():.2f})', fontsize=14)

    plt.suptitle(f'{target} Distribution', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()


from scipy.stats import boxcox, yeojohnson

def check_transformations(series, target, transforms=None, bins=50):
    """
    Menampilkan plot distribusi untuk beberapa transformasi yang dipilih.

    Parameters
    ----------
    series : pd.Series
        Data yang akan ditransformasi.
    target : str
        Nama fitur (untuk judul).
    transforms : list[str]
        List opsi transformasi.
        Supported:
            - 'original'
            - 'log'
            - 'sqrt'
            - 'boxcox'      (positif only)
            - 'yeojohnson'  (boleh negatif)
    bins : int
        Jumlah bins untuk histogram.
    """

    if transforms is None:
        transforms = ["original", "log", "sqrt"]  # default 3 transformasi utama

    valid_transforms = ["original", "log", "sqrt", "boxcox", "yeojohnson"]

    # Validasi input
    for t in transforms:
        if t not in valid_transforms:
            raise ValueError(f"Transform '{t}' belum didukung. Pilihan: {valid_transforms}")

    # Setup subplot
    fig, axes = plt.subplots(1, len(transforms), figsize=(6 * len(transforms), 5))

    # Kalau cuma 1 transformasi, axes bukan array → ubah jadi list
    if len(transforms) == 1:
        axes = [axes]

    for ax, tname in zip(axes, transforms):

        # === ORIGINAL ===
        if tname == "original":
            transformed = series.copy()
            title = "Original"

        # === LOG TRANSFORM ===
        elif tname == "log":
            transformed = np.log(series[series > 0])
            title = "Log Transform"

        # === SQRT TRANSFORM ===
        elif tname == "sqrt":
            transformed = np.sqrt(series.clip(lower=0))
            title = "Square Root"

        # === BOX-COX (Positif Only) ===
        elif tname == "boxcox":
            positive = series[series > 0]
            transformed, _ = boxcox(positive)
            title = "Box-Cox Transform"

        # === YEO JOHNSON (Dukungan Negatif) ===
        elif tname == "yeojohnson":
            transformed, _ = yeojohnson(series)
            title = "Yeo-Johnson Transform"

        # === PLOTTING ===
        sns.histplot(transformed, bins=bins, kde=True, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f"{title}\nSkew: {pd.Series(transformed).skew():.2f}", fontsize=14)
        ax.set_xlabel(target)

    plt.suptitle(f"{target} Distribution & Applied Transformations", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()



# ======================= ASTRONOMICAL FUNCTIONS ======================= #

def plot_monthly_seasonality(df, target, ax=None):
    """
    Plot monthly seasonality of energy output and GHI.
    """
    data = df.copy()
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Month'] = data['Timestamp'].dt.month

    monthly_stats = data.groupby('Month')[[target, 'GHI']].mean()

    # Axis handling
    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        standalone = True

    ax1 = ax
    ax2 = ax1.twinx()

    # Main line
    l1 = sns.lineplot(
        x=monthly_stats.index, y=monthly_stats[target],
        ax=ax1, color='blue', marker='o', label=target
    )

    # Secondary line
    l2 = sns.lineplot(
        x=monthly_stats.index, y=monthly_stats['GHI'],
        ax=ax2, color='orange', linestyle='--', marker='s', label='GHI'
    )

    # Titles & labels
    ax1.set_title("Monthly Average: Seasonality Check")
    ax1.set_ylabel(target)
    ax2.set_ylabel("GHI")
    ax1.grid(True, alpha=0.3)

    # -----------------------------------------------
    # 🔥 FIX LEGEND: ambil legend dari kedua axis
    # -----------------------------------------------
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Gabungkan
    handles = handles1 + handles2
    labels = labels1 + labels2

    # Hapus legend default → taruh gabungan di axis kiri
    ax1.legend(handles, labels, loc='upper left')

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_diurnal_cycle(df, target, ax=None):
    """
    Plot normalized diurnal cycle for TARGET, GHI, and DNI.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing Timestamp, target, GHI, DNI.
    target : str
        Target column name.
    ax : matplotlib.axes.Axes or None
        If provided, plot in subplot. Otherwise create standalone figure.
    """
    data = df.copy()
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Hour'] = data['Timestamp'].dt.hour

    hourly_stats = data.groupby('Hour')[[target, 'GHI', 'DNI']].mean()
    hourly_norm = (hourly_stats - hourly_stats.min()) / (hourly_stats.max() - hourly_stats.min())

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        standalone = True

    sns.lineplot(data=hourly_norm, x=hourly_norm.index, y=target, ax=ax, label=target)
    sns.lineplot(data=hourly_norm, x=hourly_norm.index, y='GHI', ax=ax, label='GHI')
    sns.lineplot(data=hourly_norm, x=hourly_norm.index, y='DNI', ax=ax, label='DNI', linestyle=':')

    peak_hour = hourly_stats['GHI'].idxmax()
    ax.axvline(peak_hour, color='red', linestyle='--', label=f"Peak {peak_hour}:00")

    ax.set_title("Diurnal Cycle – Solar Noon Check")
    ax.set_ylabel("Normalized Value (0-1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_physics_correlation(df, target, features, ax=None):
    """
    Plot correlation heatmap between target and physics-related features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing target + features.
    target : str
        Target column name.
    features : list
        List of feature column names (e.g. GHI, DNI, tempC).
    ax : matplotlib.axes.Axes or None
    """
    corr_cols = [target] + features
    corr_matrix = df[corr_cols].corr()

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        standalone = True

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Physics Correlation Matrix")

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_irradiance_scatter(df, target, ax=None, colors=None):
    """
    Scatter plot: GHI and DNI vs Target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing GHI, DNI, and target.
    target : str
        Target column name.
    ax : matplotlib.axes.Axes or None
    colors : list[str] or None
        Warna untuk [GHI, DNI]. Default: biru & merah kontras.
    """
    # default warna kontras
    if colors is None:
        colors = ['#1f77b4', '#d62728']   # blue, red

    data = df[df['GHI'] > 10].sample(min(2000, len(df)), random_state=42)

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        standalone = True

    sns.scatterplot(
        data=data,
        x='GHI', y=target,
        ax=ax,
        alpha=0.30,
        label='GHI vs Target',
        color=colors[0],
    )

    sns.scatterplot(
        data=data,
        x='DNI', y=target,
        ax=ax,
        alpha=0.30,
        label='DNI vs Target',
        color=colors[1],
    )

    ax.set_title("Irradiance Impact: GHI & DNI vs Energy Output")
    ax.set_xlabel("Irradiance (W/m²)")
    ax.set_ylabel(target)
    ax.legend()

    if standalone:
        plt.tight_layout()
        plt.show()


def _prepare_month(df):
    """Internal helper: add Month_Name column with correct order."""
    month_map = {
        1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
        7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'
    }
    order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Month'] = df['Timestamp'].dt.month
    df['Month_Name'] = df['Month'].map(month_map)

    return df, order


def plot_monthly_target_distribution(df, target_col='% Baseline', ax=None):
    """
    Monthly distribution plot for target (% Baseline).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing Timestamp and target column.
    target_col : str
        Column name of the target variable.
    ax : matplotlib.axes.Axes or None
        If provided, plot within subplot; if None, create standalone figure.
    """
    data, order = _prepare_month(df)

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
        standalone = True

    sns.boxplot(
        data=data, x='Month_Name', y=target_col,
        order=order, palette='viridis', ax=ax
    )
    ax.set_title('Monthly Energy Distribution (Target)', fontsize=14)
    ax.set_ylabel('Energy Output (%)')
    ax.grid(True, axis='y', alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_monthly_ghi_distribution(df, ax=None):
    """
    Monthly GHI distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing Timestamp and GHI.
    ax : matplotlib.axes.Axes or None
        If provided, plot within subplot; if None, create standalone figure.
    """
    data, order = _prepare_month(df)

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
        standalone = True

    sns.boxplot(
        data=data, x='Month_Name', y='GHI',
        order=order, palette='Oranges', ax=ax
    )
    ax.set_title('Monthly Irradiance Distribution (GHI)', fontsize=14)
    ax.set_ylabel('GHI (W/m²)')
    ax.grid(True, axis='y', alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_monthly_temperature_distribution(df, ax=None):
    """
    Monthly temperature distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing Timestamp and tempC.
    ax : matplotlib.axes.Axes or None
        If provided, plot within subplot; if None, create standalone figure.
    """
    data, order = _prepare_month(df)

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
        standalone = True

    sns.boxplot(
        data=data, x='Month_Name', y='tempC',
        order=order, palette='coolwarm', ax=ax
    )
    ax.set_title('Monthly Temperature Distribution', fontsize=14)
    ax.set_ylabel('Temperature (°C)')
    ax.grid(True, axis='y', alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()


def _prep_daylight(df):
    """Return two filtered datasets: day_data and high_sun_data."""
    day_data = df[df['GHI'] > 10].copy()
    high_sun_data = df[df['GHI'] > 200].copy()
    high_sun_data['raw_efficiency'] = high_sun_data['% Baseline'] / high_sun_data['GHI']
    return day_data, high_sun_data


def plot_temp_vs_efficiency(df, ax=None):
    """
    Analyze thermal effects: Temperature vs raw panel efficiency.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing tempC, GHI, and % Baseline.
    ax : matplotlib.axes.Axes or None
        If provided, plot inside subplot; otherwise standalone.
    """
    day_data, high_sun_data = _prep_daylight(df)

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        standalone = True

    sns.scatterplot(
        data=high_sun_data,
        x='tempC', y='raw_efficiency',
        alpha=0.1, ax=ax, color='crimson'
    )
    sns.regplot(
        data=high_sun_data,
        x='tempC', y='raw_efficiency',
        scatter=False, ax=ax,
        color='black', line_kws={'linestyle': '--'}
    )

    ax.set_title('Thermal Physics: Temperature vs Efficiency')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Raw Efficiency')
    ax.grid(True, alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_cloudcover_vs_energy(df, ax=None):
    """
    Cloud opacity impact: Cloud Cover (%) vs Energy Output.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing cloudcover, % Baseline, GHI.
    ax : matplotlib.axes.Axes or None
    """
    day_data, _ = _prep_daylight(df)

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        standalone = True

    sns.scatterplot(
        data=day_data,
        x='cloudcover', y='% Baseline',
        alpha=0.05, ax=ax, color='dodgerblue'
    )
    sns.regplot(
        data=day_data,
        x='cloudcover', y='% Baseline',
        scatter=False, ax=ax, color='darkblue'
    )

    ax.set_title('Cloud Opacity: Cloud Cover vs Energy Output')
    ax.set_xlabel('Cloud Cover (%)')
    ax.set_ylabel('Energy Output (%)')
    ax.grid(True, alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_cloudtype_impact(df, sort_by='median', ax=None):
    """
    Impact of cloud categories on energy output.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing Cloud Type, % Baseline, GHI.
    sort_by : {'median', 'Q3'}
        Sorting method for categories.
    ax : matplotlib.axes.Axes or None
    """
    day_data, _ = _prep_daylight(df)

    if sort_by == 'Q3':
        order_metric = day_data.groupby('Cloud Type')['% Baseline'].quantile(0.75)
    else:
        order_metric = day_data.groupby('Cloud Type')['% Baseline'].median()

    order_sorted = order_metric.sort_values(ascending=False).index.tolist()

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        standalone = True

    sns.boxplot(
        data=day_data,
        x='Cloud Type', y='% Baseline',
        order=order_sorted, ax=ax, palette='Spectral'
    )
    ax.set_title(f'Cloud Type Impact (Sorted by {sort_by})')
    ax.set_xlabel('Cloud Type')
    ax.set_ylabel('Energy Output (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    if standalone:
        plt.tight_layout()
        plt.show()