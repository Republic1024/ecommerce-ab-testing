"""
Marketing A/B Testing: Advertisement vs PSA Effectiveness Analysis
================================================================

Business Question: Does showing advertisements instead of PSA (Public Service Announcements)
increase user conversion rates and engagement?

"""

# Import required libraries
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, norm
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend to avoid PyCharm conflicts
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Set visualization style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')

sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.1)

# Configure matplotlib for better compatibility
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True

print("=" * 80)
print("MARKETING A/B TESTING ANALYSIS".center(80))
print("Advertisement vs PSA Conversion Impact Study".center(80))
print("=" * 80)

# ============================================================================
# SECTION 1: DATA LOADING AND EXPLORATION
# ============================================================================

print("\n[STEP 1] Loading and Exploring Data...")

# Load the dataset
df = pd.read_csv("marketing_AB.csv")
print(f"✓ Dataset loaded successfully")
print(f"✓ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Display basic information
print(f"\n📊 Dataset Overview:")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:")
print(df.dtypes)

# Clean column names (remove spaces and standardize)
df.columns = ['index', 'user_id', 'test_group', 'converted', 'total_ads', 'most_ads_day', 'most_ads_hour']

# Data quality assessment
print(f"\n🔍 Data Quality Assessment:")
print(f"✓ Missing values: {df.isnull().sum().sum():,}")
print(f"✓ Duplicate users: {df['user_id'].duplicated().sum():,}")
print(f"✓ Unique users: {df['user_id'].nunique():,}")
print(f"✓ Total records: {len(df):,}")

# Check test groups
print(f"\n👥 Test Group Distribution:")
group_counts = df['test_group'].value_counts()
for group, count in group_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   • {group}: {count:,} users ({percentage:.1f}%)")

# Check conversion distribution
print(f"\n📈 Conversion Distribution:")
conversion_counts = df['converted'].value_counts()
total_users = len(df)
converted_users = conversion_counts.get(True, 0)
non_converted_users = conversion_counts.get(False, 0)
overall_conversion_rate = (converted_users / total_users) * 100

print(f"   • Converted: {converted_users:,} users ({(converted_users / total_users) * 100:.2f}%)")
print(f"   • Not Converted: {non_converted_users:,} users ({(non_converted_users / total_users) * 100:.2f}%)")
print(f"   • Overall Conversion Rate: {overall_conversion_rate:.2f}%")

# Remove the index column as it's not needed
df = df.drop('index', axis=1)

# ============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n[STEP 2] Exploratory Data Analysis...")

# Summary statistics by group
print(f"\n📊 Summary Statistics by Test Group:")
summary_stats = df.groupby('test_group').agg({
    'user_id': 'count',
    'converted': ['sum', 'mean'],
    'total_ads': ['mean', 'std', 'median'],
    'most_ads_hour': ['mean', 'std']
}).round(4)

print(summary_stats)

# Calculate conversion rates by group
conversion_by_group = df.groupby('test_group')['converted'].agg(['count', 'sum', 'mean']).round(4)
conversion_by_group.columns = ['total_users', 'conversions', 'conversion_rate']
print(f"\n📈 Conversion Rates by Group:")
print(conversion_by_group)

# Analyze ad exposure patterns
print(f"\n📺 Ad Exposure Analysis:")
ad_stats = df.groupby('test_group')['total_ads'].describe().round(2)
print(ad_stats)

# Time pattern analysis
print(f"\n🕐 Peak Hour Analysis:")
hour_analysis = df.groupby(['test_group', 'most_ads_hour'])['converted'].mean().unstack(fill_value=0)
print(f"Average conversion rate by peak hour (top 5 hours):")
if len(hour_analysis.columns) > 0:
    for hour in sorted(hour_analysis.columns)[:5]:
        print(f"   Hour {hour}: ad={hour_analysis.loc['ad', hour]:.3f}, psa={hour_analysis.loc['psa', hour]:.3f}")

# ============================================================================
# SECTION 3: STATISTICAL POWER ANALYSIS
# ============================================================================

print("\n[STEP 3] Statistical Power Analysis...")

# Calculate observed effect size
control_rate = df[df['test_group'] == 'psa']['converted'].mean()
treatment_rate = df[df['test_group'] == 'ad']['converted'].mean()
observed_effect = treatment_rate - control_rate
relative_effect = (observed_effect / control_rate) * 100 if control_rate > 0 else 0

print(f"🎯 Observed Effect Sizes:")
print(f"   • Control (PSA) conversion rate: {control_rate:.4f} ({control_rate * 100:.2f}%)")
print(f"   • Treatment (Ad) conversion rate: {treatment_rate:.4f} ({treatment_rate * 100:.2f}%)")
print(f"   • Absolute effect: {observed_effect:.4f}")
print(f"   • Relative effect: {relative_effect:+.2f}%")


# Power calculation for observed effect
def calculate_power_for_proportions(n1, n2, p1, p2, alpha=0.05):
    """Calculate statistical power for two-proportion test"""
    pooled_p = (n1 * p1 + n2 * p2) / (n1 + n2)
    pooled_se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))
    effect_se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = (abs(p1 - p2) - z_alpha * pooled_se) / effect_se
    power = stats.norm.cdf(z_beta)
    return power


# Calculate power for current sample sizes
n_control = len(df[df['test_group'] == 'psa'])
n_treatment = len(df[df['test_group'] == 'ad'])
current_power = calculate_power_for_proportions(n_control, n_treatment, control_rate, treatment_rate)

print(f"\n⚡ Statistical Power Analysis:")
print(f"   • Sample size - Control: {n_control:,}")
print(f"   • Sample size - Treatment: {n_treatment:,}")
print(f"   • Statistical power: {current_power:.3f} ({current_power * 100:.1f}%)")
print(f"   • Power assessment: {'Adequate (>0.8)' if current_power > 0.8 else 'Low (<0.8)'}")

# ============================================================================
# SECTION 4: HYPOTHESIS TESTING
# ============================================================================

print("\n[STEP 4] Statistical Hypothesis Testing...")


def perform_conversion_ab_test(df, control_group, treatment_group, alpha=0.05):
    """
    Perform comprehensive A/B test for conversion rates
    """
    print(f"\n🧪 A/B Test: {treatment_group.upper()} vs {control_group.upper()}")
    print("=" * 60)

    # Extract data for each group
    control_data = df[df['test_group'] == control_group]['converted']
    treatment_data = df[df['test_group'] == treatment_group]['converted']

    # Basic statistics
    control_n = len(control_data)
    treatment_n = len(treatment_data)
    control_conversions = control_data.sum()
    treatment_conversions = treatment_data.sum()
    control_rate = control_data.mean()
    treatment_rate = treatment_data.mean()

    print(f"Control Group ({control_group}):")
    print(f"   • Sample size: {control_n:,}")
    print(f"   • Conversions: {control_conversions:,}")
    print(f"   • Conversion rate: {control_rate:.4f} ({control_rate * 100:.2f}%)")

    print(f"\nTreatment Group ({treatment_group}):")
    print(f"   • Sample size: {treatment_n:,}")
    print(f"   • Conversions: {treatment_conversions:,}")
    print(f"   • Conversion rate: {treatment_rate:.4f} ({treatment_rate * 100:.2f}%)")

    # Effect size calculations
    absolute_effect = treatment_rate - control_rate
    relative_effect = (absolute_effect / control_rate) * 100 if control_rate > 0 else 0

    print(f"\nEffect Size:")
    print(f"   • Absolute difference: {absolute_effect:+.4f}")
    print(f"   • Relative improvement: {relative_effect:+.2f}%")

    # Statistical tests
    # 1. Two-proportion z-test
    count = np.array([treatment_conversions, control_conversions])
    nobs = np.array([treatment_n, control_n])
    z_stat, z_pvalue = proportions_ztest(count, nobs)

    # 2. Chi-square test
    contingency_table = pd.crosstab(df['test_group'], df['converted'])
    chi2_stat, chi2_pvalue, dof, expected = chi2_contingency(contingency_table)

    # 3. Fisher's exact test (for small samples)
    from scipy.stats import fisher_exact
    if control_conversions < 10 or treatment_conversions < 10:
        oddsratio, fisher_pvalue = fisher_exact([[treatment_conversions, treatment_n - treatment_conversions],
                                                 [control_conversions, control_n - control_conversions]])
        print(f"\nStatistical Tests:")
        print(f"   • Two-proportion z-test: z={z_stat:.3f}, p={z_pvalue:.4f}")
        print(f"   • Chi-square test: χ²={chi2_stat:.3f}, p={chi2_pvalue:.4f}")
        print(f"   • Fisher's exact test: p={fisher_pvalue:.4f} (recommended for small samples)")
        primary_pvalue = fisher_pvalue
    else:
        print(f"\nStatistical Tests:")
        print(f"   • Two-proportion z-test: z={z_stat:.3f}, p={z_pvalue:.4f}")
        print(f"   • Chi-square test: χ²={chi2_stat:.3f}, p={chi2_pvalue:.4f}")
        primary_pvalue = z_pvalue

    # Confidence interval for difference in proportions
    se_diff = np.sqrt(
        control_rate * (1 - control_rate) / control_n + treatment_rate * (1 - treatment_rate) / treatment_n)
    z_critical = stats.norm.ppf(1 - alpha / 2)
    ci_lower = absolute_effect - z_critical * se_diff
    ci_upper = absolute_effect + z_critical * se_diff

    print(f"   • 95% CI for difference: [{ci_lower:+.4f}, {ci_upper:+.4f}]")

    # Statistical conclusion
    is_significant = primary_pvalue < alpha
    print(f"\n📊 Statistical Conclusion:")
    print(f"   • Result: {'SIGNIFICANT' if is_significant else 'NOT SIGNIFICANT'} at α={alpha}")
    print(f"   • Primary p-value: {primary_pvalue:.4f}")

    if is_significant:
        direction = "INCREASE" if absolute_effect > 0 else "DECREASE"
        print(f"   • Treatment shows a significant {direction} in conversion rate")
        print(f"   • Effect magnitude: {abs(relative_effect):.1f}% relative change")
    else:
        print(f"   • No statistically significant difference detected")
        print(f"   • This could indicate: no real effect, insufficient sample size, or high variance")

    return {
        'control_group': control_group,
        'treatment_group': treatment_group,
        'control_rate': control_rate,
        'treatment_rate': treatment_rate,
        'absolute_effect': absolute_effect,
        'relative_effect': relative_effect,
        'p_value': primary_pvalue,
        'is_significant': is_significant,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'sample_sizes': {'control': control_n, 'treatment': treatment_n}
    }


# Perform the main A/B test
main_results = perform_conversion_ab_test(df, 'psa', 'ad')

# ============================================================================
# SECTION 5: SEGMENTATION ANALYSIS
# ============================================================================

print("\n[STEP 5] Segmentation Analysis...")

# Analyze by ad exposure level
print(f"\n📺 Analysis by Ad Exposure Level:")
df['ad_exposure_level'] = pd.cut(df['total_ads'],
                                 bins=[0, 5, 10, 20, float('inf')],
                                 labels=['Low (1-5)', 'Medium (6-10)', 'High (11-20)', 'Very High (21+)'])

segment_analysis = df.groupby(['test_group', 'ad_exposure_level'])['converted'].agg(['count', 'mean']).round(4)
print(segment_analysis)

# Peak hour segmentation
print(f"\n🕐 Analysis by Peak Hour Segments:")


# Create time segments
def categorize_hour(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'


df['time_segment'] = df['most_ads_hour'].apply(categorize_hour)
time_analysis = df.groupby(['test_group', 'time_segment'])['converted'].agg(['count', 'mean']).round(4)
print(time_analysis)

# Statistical test for each segment
print(f"\n🔬 Segmented Statistical Tests:")
for exposure_level in df['ad_exposure_level'].cat.categories:
    segment_data = df[df['ad_exposure_level'] == exposure_level]
    if len(segment_data) > 50 and len(segment_data['test_group'].unique()) == 2:  # Minimum sample size check
        control_segment = segment_data[segment_data['test_group'] == 'psa']['converted']
        treatment_segment = segment_data[segment_data['test_group'] == 'ad']['converted']

        if len(control_segment) > 10 and len(treatment_segment) > 10:
            # Two-proportion test for segment
            count = np.array([treatment_segment.sum(), control_segment.sum()])
            nobs = np.array([len(treatment_segment), len(control_segment)])
            z_stat, p_val = proportions_ztest(count, nobs)

            effect = treatment_segment.mean() - control_segment.mean()
            rel_effect = (effect / control_segment.mean()) * 100 if control_segment.mean() > 0 else 0

            significance = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
            print(f"   {exposure_level}: {effect:+.4f} ({rel_effect:+.1f}%), p={p_val:.4f} - {significance}")

# ============================================================================
# SECTION 6: BUSINESS IMPACT ANALYSIS
# ============================================================================

print("\n[STEP 6] Business Impact Analysis...")

# Calculate business metrics
total_users = len(df)
control_users = len(df[df['test_group'] == 'psa'])
treatment_users = len(df[df['test_group'] == 'ad'])

current_conversions_control = df[df['test_group'] == 'psa']['converted'].sum()
current_conversions_treatment = df[df['test_group'] == 'ad']['converted'].sum()

print(f"💼 Current Performance Metrics:")
print(f"   • Total users in experiment: {total_users:,}")
print(f"   • Control group conversions: {current_conversions_control:,}")
print(f"   • Treatment group conversions: {current_conversions_treatment:,}")
print(f"   • Total conversions: {current_conversions_control + current_conversions_treatment:,}")

# Project full rollout impact
if main_results['is_significant'] and main_results['absolute_effect'] > 0:
    # Assume we roll out to all users
    projected_baseline_conversions = total_users * main_results['control_rate']
    projected_treatment_conversions = total_users * main_results['treatment_rate']
    additional_conversions = projected_treatment_conversions - projected_baseline_conversions

    print(f"\n🚀 Projected Full Rollout Impact:")
    print(f"   • Baseline conversions (PSA): {projected_baseline_conversions:.0f}")
    print(f"   • Projected conversions (Ads): {projected_treatment_conversions:.0f}")
    print(f"   • Additional conversions: {additional_conversions:.0f}")
    print(f"   • Relative improvement: {main_results['relative_effect']:.1f}%")

    # Revenue impact (assuming conversion value)
    assumed_conversion_value = 25  # Assume $25 per conversion
    revenue_impact = additional_conversions * assumed_conversion_value
    print(f"   • Estimated revenue impact: ${revenue_impact:,.0f}")
    print(f"     (assuming ${assumed_conversion_value} per conversion)")

else:
    print(f"\n❌ No Significant Business Impact Detected:")
    print(f"   • Current results do not support full rollout")
    print(f"   • Consider: longer test period, larger sample, or different approach")

# Cost-benefit analysis
print(f"\n💰 Cost-Benefit Considerations:")
print(f"   • Ad campaign costs: Variable (need actual data)")
print(f"   • PSA opportunity cost: Minimal")
print(f"   • Implementation complexity: Low")
print(f"   • Risk assessment: {'Low' if main_results['is_significant'] else 'Medium-High'}")

# ============================================================================
# SECTION 7: VISUALIZATIONS
# ============================================================================

print("\n[STEP 7] Creating Visualizations...")

try:
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Marketing A/B Test Analysis: Ads vs PSA', fontsize=16, fontweight='bold')

    # 1. Conversion rates by group
    groups = ['PSA', 'Ad']
    rates = [main_results['control_rate'], main_results['treatment_rate']]
    colors = ['lightblue', 'lightcoral']

    bars = axes[0, 0].bar(groups, rates, color=colors, alpha=0.8)
    axes[0, 0].set_title('Conversion Rates by Test Group')
    axes[0, 0].set_ylabel('Conversion Rate')
    axes[0, 0].set_ylim(0, max(rates) * 1.2)

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                        f'{rate:.3f}\n({rate * 100:.1f}%)', ha='center', va='bottom', fontweight='bold')

    # 2. Sample sizes
    sample_sizes = [control_users, treatment_users]
    axes[0, 1].bar(groups, sample_sizes, color=colors, alpha=0.8)
    axes[0, 1].set_title('Sample Sizes by Group')
    axes[0, 1].set_ylabel('Number of Users')

    for bar, size in zip(axes[0, 1].patches, sample_sizes):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                        f'{size:,}', ha='center', va='bottom', fontweight='bold')

    # 3. Effect size with confidence interval
    effect = main_results['absolute_effect']
    ci_lower = main_results['ci_lower']
    ci_upper = main_results['ci_upper']

    axes[0, 2].bar(['Effect Size'], [effect], color='green' if effect > 0 else 'red', alpha=0.7)
    axes[0, 2].errorbar(['Effect Size'], [effect], yerr=[[effect - ci_lower], [ci_upper - effect]],
                        fmt='none', color='black', capsize=5)
    axes[0, 2].set_title('Treatment Effect with 95% CI')
    axes[0, 2].set_ylabel('Conversion Rate Difference')
    axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 2].text(0, effect + (ci_upper - effect) / 2, f'{effect:.4f}\n±{(ci_upper - effect):.4f}',
                    ha='center', va='bottom', fontweight='bold')

    # 4. Ad exposure distribution
    df.boxplot(column='total_ads', by='test_group', ax=axes[1, 0])
    axes[1, 0].set_title('Ad Exposure Distribution by Group')
    axes[1, 0].set_xlabel('Test Group')
    axes[1, 0].set_ylabel('Total Ads Seen')

    # 5. Conversion by time segment
    time_conv = df.groupby(['test_group', 'time_segment'])['converted'].mean().unstack()
    time_conv.plot(kind='bar', ax=axes[1, 1], color=['lightblue', 'lightcoral'])
    axes[1, 1].set_title('Conversion Rate by Time Segment')
    axes[1, 1].set_ylabel('Conversion Rate')
    axes[1, 1].set_xlabel('Test Group')
    axes[1, 1].legend(title='Time Segment')
    axes[1, 1].tick_params(axis='x', rotation=0)

    # 6. Statistical significance visualization
    p_value = main_results['p_value']
    significance_colors = ['green' if p_value < 0.05 else 'orange' if p_value < 0.1 else 'red']
    significance_labels = ['Significant (p<0.05)' if p_value < 0.05 else
                           'Marginally Significant (0.05≤p<0.1)' if p_value < 0.1 else
                           'Not Significant (p≥0.1)']

    axes[1, 2].bar(['P-value'], [-np.log10(p_value)], color=significance_colors[0], alpha=0.7)
    axes[1, 2].axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='α=0.05')
    axes[1, 2].axhline(y=-np.log10(0.1), color='orange', linestyle='--', alpha=0.7, label='α=0.10')
    axes[1, 2].set_title('Statistical Significance')
    axes[1, 2].set_ylabel('-log10(p-value)')
    axes[1, 2].legend()
    axes[1, 2].text(0, -np.log10(p_value) + 0.1, f'p={p_value:.4f}\n{significance_labels[0]}',
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('marketing_ab_test_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Visualizations saved as 'marketing_ab_test_analysis.png'")

except Exception as e:
    print(f"⚠️ Visualization error: {str(e)}")

plt.close()

# ============================================================================
# SECTION 8: CONCLUSIONS AND RECOMMENDATIONS
# ============================================================================

print("\n[STEP 8] Conclusions and Recommendations...")
print("=" * 80)

print(f"\n🎯 EXECUTIVE SUMMARY:")
print(f"Test Duration: Full dataset analysis")
print(f"Sample Size: {total_users:,} total users ({control_users:,} control, {treatment_users:,} treatment)")
print(f"Primary Metric: Conversion rate")
print(f"Statistical Method: Two-proportion z-test with 95% confidence interval")

print(f"\n📊 KEY FINDINGS:")

# Statistical significance
if main_results['is_significant']:
    print(f"✅ STATISTICALLY SIGNIFICANT RESULT (p={main_results['p_value']:.4f})")
    direction = "INCREASE" if main_results['absolute_effect'] > 0 else "DECREASE"
    print(f"✅ Ads show a significant {direction} in conversion rate")
    print(
        f"✅ Effect size: {main_results['absolute_effect']:+.4f} absolute ({main_results['relative_effect']:+.1f}% relative)")
else:
    print(f"❌ NO STATISTICALLY SIGNIFICANT DIFFERENCE (p={main_results['p_value']:.4f})")
    print(f"❌ Cannot conclude that ads perform differently than PSA")

# Effect size interpretation
effect_magnitude = abs(main_results['relative_effect'])
if effect_magnitude < 5:
    effect_interpretation = "Small"
elif effect_magnitude < 20:
    effect_interpretation = "Medium"
else:
    effect_interpretation = "Large"

print(f"📏 Effect Size: {effect_interpretation} ({effect_magnitude:.1f}% relative change)")

# Confidence interval interpretation
ci_contains_zero = main_results['ci_lower'] <= 0 <= main_results['ci_upper']
print(f"🎯 95% Confidence Interval: [{main_results['ci_lower']:+.4f}, {main_results['ci_upper']:+.4f}]")
if ci_contains_zero and not main_results['is_significant']:
    print("   • Interval contains zero - consistent with no effect")
elif not ci_contains_zero and main_results['is_significant']:
    print("   • Interval does not contain zero - consistent with real effect")

print(f"\n💡 BUSINESS RECOMMENDATIONS:")

if main_results['is_significant'] and main_results['absolute_effect'] > 0:
    print(f"🚀 PRIMARY RECOMMENDATION: IMPLEMENT ADS STRATEGY")
    print(f"   • Statistical evidence supports ad effectiveness over PSA")
    print(f"   • Projected conversion improvement: {main_results['relative_effect']:.1f}%")
    print(f"   • Implementation confidence: High")

    print(f"\n📋 IMPLEMENTATION PLAN:")
    print(f"   Phase 1: Gradual rollout to 25% of users (Month 1)")
    print(f"   Phase 2: Monitor performance and expand to 50% (Month 2)")
    print(f"   Phase 3: Full implementation with continued monitoring (Month 3+)")

    print(f"\n🎯 SUCCESS METRICS TO MONITOR:")
    print(f"   • Primary: Conversion rate maintenance above {main_results['treatment_rate']:.3f}")
    print(f"   • Secondary: User engagement, ad fatigue, cost per conversion")
    print(f"   • Risk indicators: Significant drop in conversion rate or user satisfaction")

elif main_results['p_value'] < 0.1:  # Marginally significant
    print(f"🔄 CONDITIONAL RECOMMENDATION: EXTENDED TESTING")
    print(f"   • Results are marginally significant (p={main_results['p_value']:.3f})")
    print(f"   • Extend test duration or increase sample size")
    print(f"   • Current effect suggests potential: {main_results['relative_effect']:+.1f}%")

else:
    print(f"❌ RECOMMENDATION: DO NOT IMPLEMENT")
    print(f"   • No statistical evidence of ad superiority over PSA")
    print(f"   • Consider alternative strategies:")
    print(f"     - Different ad creative or messaging")
    print(f"     - Targeted audience segments")
    print(f"     - Alternative conversion optimization approaches")

print(f"\n⚠️ LIMITATIONS & CONSIDERATIONS:")
print(f"   • Sample Bias: Results may not generalize to all user segments")
print(f"   • Temporal Effects: Seasonal or time-based variations not fully captured")
print(f"   • Ad Fatigue: Long-term effectiveness may decline")
print(f"   • Cost Considerations: Ad spend vs PSA opportunity cost not quantified")
print(f"   • External Validity: Results specific to current user base and context")

print(f"\n🔬 STATISTICAL ROBUSTNESS:")
power_assessment = "Adequate" if current_power > 0.8 else "Limited"
print(f"   • Statistical Power: {current_power:.3f} ({power_assessment})")
print(f"   • Sample Size: {total_users:,} users (large sample)")
print(f"   • Effect Size: {effect_interpretation} practical significance")
print(f"   • Multiple Testing: Single primary hypothesis (low risk)")

print(f"\n📈 NEXT STEPS:")
if main_results['is_significant']:
    print(f"   1. Prepare implementation plan and resource allocation")
    print(f"   2. Set up monitoring dashboards for key metrics")
    print(f"   3. Plan A/B test for ad creative optimization")
    print(f"   4. Establish rollback procedures if performance degrades")
else:
    print(f"   1. Analyze user segments to identify responsive populations")
    print(f"   2. Test alternative ad formats or messaging")
    print(f"   3. Consider increasing sample size for more statistical power")
    print(f"   4. Investigate confounding variables or alternative hypotheses")

print(f"\n💰 COST-BENEFIT SUMMARY:")
if main_results['is_significant'] and main_results['absolute_effect'] > 0:
    # Estimate potential revenue impact
    monthly_users = 100000  # Assumed monthly user base
    conversion_lift = main_results['absolute_effect']
    additional_monthly_conversions = monthly_users * conversion_lift
    assumed_ltv_per_conversion = 50  # Assumed lifetime value per conversion
    monthly_revenue_impact = additional_monthly_conversions * assumed_ltv_per_conversion
    annual_revenue_impact = monthly_revenue_impact * 12

    print(f"   • Estimated additional monthly conversions: {additional_monthly_conversions:.0f}")
    print(f"   • Estimated monthly revenue impact: ${monthly_revenue_impact:,.0f}")
    print(f"   • Estimated annual revenue impact: ${annual_revenue_impact:,.0f}")
    print(f"   • ROI Assessment: Positive (assuming ad costs < revenue uplift)")

else:
    print(f"   • No quantifiable revenue impact from current results")
    print(f"   • Consider cost of extended testing vs opportunity cost")

# Export detailed results
results_summary = pd.DataFrame([{
    'Metric': 'Conversion Rate',
    'Control_Group': main_results['control_group'],
    'Treatment_Group': main_results['treatment_group'],
    'Control_Rate': main_results['control_rate'],
    'Treatment_Rate': main_results['treatment_rate'],
    'Absolute_Effect': main_results['absolute_effect'],
    'Relative_Effect_Pct': main_results['relative_effect'],
    'P_Value': main_results['p_value'],
    'Is_Significant': main_results['is_significant'],
    'CI_Lower': main_results['ci_lower'],
    'CI_Upper': main_results['ci_upper'],
    'Control_Sample_Size': main_results['sample_sizes']['control'],
    'Treatment_Sample_Size': main_results['sample_sizes']['treatment'],
    'Statistical_Power': current_power
}])

print(f"\n📊 DETAILED RESULTS SUMMARY:")
print("=" * 80)
for col in ['Control_Rate', 'Treatment_Rate', 'Absolute_Effect', 'Relative_Effect_Pct', 'P_Value']:
    value = results_summary[col].iloc[0]
    if col == 'P_Value':
        print(f"{col.replace('_', ' ')}: {value:.6f}")
    elif col == 'Relative_Effect_Pct':
        print(f"{col.replace('_', ' ')}: {value:+.2f}%")
    else:
        print(f"{col.replace('_', ' ')}: {value:.6f}")

print(f"Statistical Significance: {'YES' if main_results['is_significant'] else 'NO'}")
print(f"Confidence Interval (95%): [{main_results['ci_lower']:+.6f}, {main_results['ci_upper']:+.6f}]")
print(f"Statistical Power: {current_power:.3f}")

# Save results to CSV
try:
    results_summary.to_csv('marketing_ab_test_results.csv', index=False)

    # Save detailed data analysis
    detailed_analysis = df.groupby('test_group').agg({
        'user_id': 'count',
        'converted': ['sum', 'mean', 'std'],
        'total_ads': ['mean', 'median', 'std'],
        'most_ads_hour': ['mean', 'std']
    }).round(6)
    detailed_analysis.to_csv('detailed_group_analysis.csv')

    print(f"\n✅ RESULTS EXPORTED:")
    print(f"   • marketing_ab_test_results.csv - Main results summary")
    print(f"   • detailed_group_analysis.csv - Detailed group statistics")

except Exception as e:
    print(f"\n⚠️ Could not save CSV files: {str(e)}")

print(f"\n📁 GENERATED FILES:")
print(f"   • marketing_ab_test_analysis.png - Comprehensive visualizations")
print(f"   • marketing_ab_test_results.csv - Statistical results")
print(f"   • detailed_group_analysis.csv - Group-level analysis")

print(f"\n🎓 METHODOLOGY VALIDATION:")
print(f"   • Experimental Design: Randomized controlled trial (assumed)")
print(f"   • Primary Analysis: Two-proportion z-test")
print(f"   • Secondary Analysis: Chi-square test")
print(f"   • Effect Size: Absolute and relative differences")
print(f"   • Confidence Intervals: Normal approximation (large sample)")
print(f"   • Multiple Comparisons: Single primary endpoint (no adjustment)")
print(f"   • Missing Data: Complete case analysis")

# Final assessment
total_effect_score = 0
if main_results['is_significant']:
    total_effect_score += 3
if abs(main_results['relative_effect']) > 10:  # >10% effect
    total_effect_score += 2
if current_power > 0.8:
    total_effect_score += 2
if main_results['ci_lower'] > 0:  # CI doesn't include zero
    total_effect_score += 2
if main_results['p_value'] < 0.01:  # Highly significant
    total_effect_score += 1

confidence_levels = {
    (8, 11): "Very High",  # Changed upper bound to 11 to include score=10
    (6, 8): "High",
    (4, 6): "Medium",
    (2, 4): "Low",
    (0, 2): "Very Low"
}

# Fixed logic to handle edge cases
confidence_level = "Very Low"  # Default
for (low, high), level in confidence_levels.items():
    if low <= total_effect_score < high:
        confidence_level = level
        break

print(f"\n🎯 OVERALL CONFIDENCE ASSESSMENT:")
print(f"   • Confidence Score: {total_effect_score}/10")
print(f"   • Confidence Level: {confidence_level}")
print(
    f"   • Recommendation Strength: {'Strong' if total_effect_score >= 6 else 'Moderate' if total_effect_score >= 4 else 'Weak'}")

print("\n" + "=" * 80)
print("MARKETING A/B TEST ANALYSIS COMPLETED SUCCESSFULLY".center(80))
print("=" * 80)

print(f"\n📞 FOR QUESTIONS OR DEEPER ANALYSIS:")
print(f"   • Review segmentation results for targeted strategies")
print(f"   • Consider longitudinal analysis for time-based effects")
print(f"   • Explore interaction effects between variables")
print(f"   • Conduct cost-effectiveness analysis with actual cost data")
print(f"   • Plan follow-up experiments based on these findings")