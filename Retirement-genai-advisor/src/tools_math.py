from langchain_core.tools import tool

@tool("calculate_retirement_growth")
def calculate_retirement_growth(current_principal: float, 
                                annual_contribution: float, 
                                years_to_grow: int, 
                                annual_return_rate: float = 0.07) -> str:
    """
    Calculates the future value of retirement savings with compound interest.
    Use this when a user asks "How much will I have in 10 years?" or "Project my savings".

    Args:
        current_principal: The current amount of money saved (e.g., 50000).
        annual_contribution: How much is added each year (e.g., 6000).
        years_to_grow: Number of years to let it grow.
        annual_return_rate: Estimated annual return (default is 0.07 for 7%).
    """
    try:
        # Compound Interest : FV = P(1+r)^t + c * [ ((1+r)^t - 1) / r ]
        rate = float(annual_return_rate)
        p = float(current_principal)
        c = float(annual_contribution)
        t = int(years_to_grow)
        
        future_value = p * ((1 + rate) ** t) + c * (((1 + rate) ** t - 1) / rate)
        
        return f"Projected Retirement Savings after {t} years: ${future_value:,.2f} (assumed {rate*100:.1f}% return)"
    except Exception as e:
        return f"Error calculating growth: {e}"
    

@tool("calculate_rmd")
def calculate_rmd(age: int, account_balance: float) -> str:
    """
    Calculates the Required Minimum Distribution (RMD) for a retirement account based on IRS Uniform Lifetime Table.
    Use this when a user asks "What is my RMD?" or "How much must I withdraw?".
    
    Args:
        age: The age of the retiree (must be 73 or older for standard RMDs).
        account_balance: The total value of tax-deferred retirement accounts.
    """
    # Simplified IRS Uniform Lifetime Table 
    # In a real app, you'd load the full CSV/JSON
    irs_distribution_periods = {
        73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
        80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 90: 12.2, 95: 8.9
    }
    
    if age < 73:
        return f"At age {age}, RMDs are generally not required. They typically begin at age 73."
    
    factor = irs_distribution_periods.get(age)
    if not factor:
        # Fallback logic for very old ages not in our mini-table
        if age > 95: factor = 8.9 
        else: factor = 27.4 # Fallback for younger
        
    rmd_amount = account_balance / factor
    return f"For age {age}, the IRS distribution period factor is {factor}. The estimated RMD is ${rmd_amount:,.2f}."