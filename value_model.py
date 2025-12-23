"""
value_model.py
Module for calculating player value metrics combining WAR and contract data
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlayerValueModel:
    """Calculate player value for trades based on WAR and contract economics"""
    
    def __init__(self, war_data: pd.DataFrame, 
                 arbitration_contracts: pd.DataFrame,
                 free_agency_contracts: pd.DataFrame):
        """
        Initialize value model
        
        Args:
            war_data: Player WAR data
            arbitration_contracts: Arbitration contract data
            free_agency_contracts: Free agency contract data
        """
        self.war_data = war_data.copy()
        self.arb_contracts = arbitration_contracts.copy()
        self.fa_contracts = free_agency_contracts.copy()
        
        logger.info("Initialized Player Value Model")
        logger.info(f"WAR records: {len(self.war_data)}")
        logger.info(f"Arbitration contracts: {len(self.arb_contracts)}")
        logger.info(f"Free agency contracts: {len(self.fa_contracts)}")
    
    def calculate_dollars_per_war(self, year: int = None) -> Dict[str, float]:
        """
        Calculate market rate ($/WAR) for arbitration and free agency
        
        Args:
            year: Specific year (if None, uses all data)
            
        Returns:
            Dictionary with dollars per WAR for each contract type
        """
        logger.info(f"Calculating $/WAR for year: {year if year else 'all'}")
        
        results = {}
        
        # Filter by year if specified
        war_df = self.war_data if year is None else self.war_data[self.war_data['year'] == year]
        arb_df = self.arb_contracts if year is None else self.arb_contracts[self.arb_contracts['year'] == year]
        fa_df = self.fa_contracts if year is None else self.fa_contracts[self.fa_contracts['year'] == year]
        
        # Arbitration $/WAR
        arb_with_war = arb_df.merge(war_df[['player_id', 'year', 'war']], 
                                     on=['player_id', 'year'], how='inner')
        arb_with_war = arb_with_war[arb_with_war['war'] > 0]
        
        if len(arb_with_war) > 0:
            arb_with_war['dollars_per_war'] = arb_with_war['salary'] / arb_with_war['war']
            results['arbitration'] = arb_with_war['dollars_per_war'].median()
        else:
            results['arbitration'] = None
        
        # Free agency $/WAR
        fa_with_war = fa_df.merge(war_df[['player_id', 'year', 'war']], 
                                   on=['player_id', 'year'], how='inner')
        fa_with_war = fa_with_war[fa_with_war['war'] > 0]
        
        if len(fa_with_war) > 0:
            fa_with_war['dollars_per_war'] = fa_with_war['salary'] / fa_with_war['war']
            results['free_agency'] = fa_with_war['dollars_per_war'].median()
        else:
            results['free_agency'] = None
        
        logger.info(f"Arbitration $/WAR: ${results['arbitration']:,.0f}" if results['arbitration'] else "N/A")
        logger.info(f"Free Agency $/WAR: ${results['free_agency']:,.0f}" if results['free_agency'] else "N/A")
        
        return results
    
    def calculate_historical_market_rates(self, start_year: int = 1985) -> pd.DataFrame:
        """
        Calculate historical $/WAR rates by year
        
        Args:
            start_year: Starting year
            
        Returns:
            DataFrame with yearly market rates
        """
        logger.info("Calculating historical market rates...")
        
        years = range(start_year, self.war_data['year'].max() + 1)
        
        rates = []
        for year in years:
            year_rates = self.calculate_dollars_per_war(year)
            rates.append({
                'year': year,
                'arb_dollars_per_war': year_rates.get('arbitration'),
                'fa_dollars_per_war': year_rates.get('free_agency')
            })
        
        df = pd.DataFrame(rates)
        
        # Calculate ratio
        df['fa_premium'] = df['fa_dollars_per_war'] / df['arb_dollars_per_war']
        
        logger.info(f"Calculated rates for {len(df)} years")
        return df
    
    def calculate_player_surplus_value(self, player_id: str, year: int, 
                                       market_rates: Dict[str, float]) -> Dict:
        """
        Calculate surplus value for a player
        
        Surplus value = Market value - Contract cost
        
        Args:
            player_id: Player identifier
            year: Season year
            market_rates: Dictionary with $/WAR rates
            
        Returns:
            Dictionary with surplus value metrics
        """
        # Get player WAR
        player_war = self.war_data[
            (self.war_data['player_id'] == player_id) & 
            (self.war_data['year'] == year)
        ]
        
        if len(player_war) == 0:
            return None
        
        war = player_war.iloc[0]['war']
        
        # Get player contract
        arb_contract = self.arb_contracts[
            (self.arb_contracts['player_id'] == player_id) & 
            (self.arb_contracts['year'] == year)
        ]
        
        fa_contract = self.fa_contracts[
            (self.fa_contracts['player_id'] == player_id) & 
            (self.fa_contracts['year'] == year)
        ]
        
        # Determine contract type and cost
        if len(arb_contract) > 0:
            contract_type = 'arbitration'
            actual_cost = arb_contract.iloc[0]['salary']
            market_rate = market_rates.get('free_agency', 8_000_000)  # Default FA rate
        elif len(fa_contract) > 0:
            contract_type = 'free_agency'
            actual_cost = fa_contract.iloc[0]['salary']
            market_rate = market_rates.get('free_agency', 8_000_000)
        else:
            contract_type = 'unknown'
            actual_cost = 0
            market_rate = market_rates.get('free_agency', 8_000_000)
        
        # Calculate values
        market_value = war * market_rate
        surplus_value = market_value - actual_cost
        
        return {
            'player_id': player_id,
            'year': year,
            'war': war,
            'contract_type': contract_type,
            'actual_cost': actual_cost,
            'market_value': market_value,
            'surplus_value': surplus_value,
            'value_ratio': market_value / actual_cost if actual_cost > 0 else np.inf
        }
    
    def calculate_trade_value_index(self, player_id: str, 
                                   projection_years: int = 3) -> Dict:
        """
        Calculate comprehensive trade value index for a player
        
        Considers:
        - Recent WAR performance
        - Contract status and costs
        - Age and career trajectory
        - Projected future value
        
        Args:
            player_id: Player identifier
            projection_years: Years to project forward
            
        Returns:
            Trade value metrics
        """
        # Get player data
        player_data = self.war_data[self.war_data['player_id'] == player_id].copy()
        
        if len(player_data) == 0:
            return None
        
        player_data = player_data.sort_values('year')
        latest_year = player_data['year'].max()
        latest_data = player_data[player_data['year'] == latest_year].iloc[0]
        
        # Recent performance (last 3 years)
        recent_years = player_data[player_data['year'] >= latest_year - 2]
        avg_recent_war = recent_years['war'].mean()
        
        # Career performance
        career_war = player_data['war'].sum()
        
        # Age factor (peak is around 27-28)
        age = latest_data.get('age', 28)
        if age <= 27:
            age_factor = 1.0 + (27 - age) * 0.02  # Younger is better
        else:
            age_factor = 1.0 - (age - 27) * 0.03  # Decline with age
        age_factor = max(0.5, min(1.2, age_factor))
        
        # Contract status
        arb_contract = self.arb_contracts[
            (self.arb_contracts['player_id'] == player_id) & 
            (self.arb_contracts['year'] == latest_year)
        ]
        fa_contract = self.fa_contracts[
            (self.fa_contracts['player_id'] == player_id) & 
            (self.fa_contracts['year'] == latest_year)
        ]
        
        if len(arb_contract) > 0:
            contract_type = 'arbitration'
            years_control = 6 - (latest_year - player_data['year'].min())  # Rough estimate
            contract_value_factor = 1.3  # Arbitration players more valuable
        elif len(fa_contract) > 0:
            contract_type = 'free_agency'
            years_control = fa_contract.iloc[0].get('contract_length', 1)
            contract_value_factor = 1.0
        else:
            contract_type = 'pre_arb'
            years_control = 6
            contract_value_factor = 1.5  # Pre-arb players most valuable
        
        # Project future WAR
        projected_war = avg_recent_war * age_factor
        projected_total_war = projected_war * min(years_control, projection_years)
        
        # Calculate trade value index (0-100 scale)
        base_value = avg_recent_war * 10  # Recent WAR worth 10 points each
        age_bonus = age_factor * 20
        control_bonus = min(years_control, 6) * 5
        contract_bonus = (contract_value_factor - 1.0) * 20
        
        trade_value_index = base_value + age_bonus + control_bonus + contract_bonus
        trade_value_index = max(0, min(100, trade_value_index))
        
        return {
            'player_id': player_id,
            'name': latest_data.get('name', 'Unknown'),
            'year': latest_year,
            'age': age,
            'recent_war_avg': avg_recent_war,
            'career_war': career_war,
            'contract_type': contract_type,
            'years_control': years_control,
            'projected_war': projected_war,
            'projected_total_war': projected_total_war,
            'trade_value_index': trade_value_index,
            'age_factor': age_factor,
            'contract_value_factor': contract_value_factor
        }
    
    def rank_players_by_trade_value(self, year: int, top_n: int = 100) -> pd.DataFrame:
        """
        Rank all players by trade value for a given year
        
        Args:
            year: Season year
            top_n: Number of top players to return
            
        Returns:
            Ranked DataFrame
        """
        logger.info(f"Ranking players by trade value for {year}...")
        
        # Get all players active in that year
        active_players = self.war_data[self.war_data['year'] == year]['player_id'].unique()
        
        trade_values = []
        for player_id in active_players:
            tv = self.calculate_trade_value_index(player_id)
            if tv:
                trade_values.append(tv)
        
        df = pd.DataFrame(trade_values)
        df = df.sort_values('trade_value_index', ascending=False).head(top_n)
        df['rank'] = range(1, len(df) + 1)
        
        logger.info(f"Ranked {len(df)} players")
        return df
    
    def compare_contract_types(self) -> pd.DataFrame:
        """
        Compare value metrics between arbitration and free agency
        
        Returns:
            Comparison DataFrame
        """
        logger.info("Comparing contract types...")
        
        # Merge WAR with contracts
        arb_value = self.arb_contracts.merge(
            self.war_data[['player_id', 'year', 'war']], 
            on=['player_id', 'year'], 
            how='inner'
        )
        arb_value = arb_value[arb_value['war'] > 0]
        arb_value['dollars_per_war'] = arb_value['salary'] / arb_value['war']
        
        fa_value = self.fa_contracts.merge(
            self.war_data[['player_id', 'year', 'war']], 
            on=['player_id', 'year'], 
            how='inner'
        )
        fa_value = fa_value[fa_value['war'] > 0]
        fa_value['dollars_per_war'] = fa_value['salary'] / fa_value['war']
        
        comparison = pd.DataFrame({
            'metric': ['count', 'avg_war', 'median_war', 'avg_salary', 
                      'median_salary', 'avg_$/war', 'median_$/war'],
            'arbitration': [
                len(arb_value),
                arb_value['war'].mean(),
                arb_value['war'].median(),
                arb_value['salary'].mean(),
                arb_value['salary'].median(),
                arb_value['dollars_per_war'].mean(),
                arb_value['dollars_per_war'].median()
            ],
            'free_agency': [
                len(fa_value),
                fa_value['war'].mean(),
                fa_value['war'].median(),
                fa_value['salary'].mean(),
                fa_value['salary'].median(),
                fa_value['dollars_per_war'].mean(),
                fa_value['dollars_per_war'].median()
            ]
        })
        
        return comparison
    
    def save_value_metrics(self, output_dir: Path):
        """
        Save all value metrics to files
        
        Args:
            output_dir: Output directory
        """
        logger.info(f"Saving value metrics to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Historical market rates
        rates = self.calculate_historical_market_rates()
        rates.to_csv(output_dir / "market_rates_history.csv", index=False)
        
        # Contract comparison
        comparison = self.compare_contract_types()
        comparison.to_csv(output_dir / "contract_type_comparison.csv", index=False)
        
        logger.info("Value metrics saved")


def main():
    """Example usage"""
    from pathlib import Path
    
    # Load data
    data_dir = Path("data/processed")
    
    try:
        war_data = pd.read_parquet(data_dir / "war_baseball_reference.parquet")
        arb_contracts = pd.read_parquet(data_dir / "contracts_arbitration.parquet")
        fa_contracts = pd.read_parquet(data_dir / "contracts_free_agency.parquet")
        
        # Create model
        model = PlayerValueModel(war_data, arb_contracts, fa_contracts)
        
        # Calculate market rates
        current_year = war_data['year'].max()
        rates = model.calculate_dollars_per_war(current_year)
        
        print(f"\nMarket Rates for {current_year}:")
        print(f"Arbitration: ${rates.get('arbitration', 0):,.0f} per WAR")
        print(f"Free Agency: ${rates.get('free_agency', 0):,.0f} per WAR")
        
        # Rank players
        top_trade_values = model.rank_players_by_trade_value(current_year, top_n=20)
        print(f"\nTop 20 Trade Values for {current_year}:")
        print(top_trade_values[['rank', 'name', 'trade_value_index', 'recent_war_avg', 'age']])
        
        # Save outputs
        output_dir = Path("outputs")
        model.save_value_metrics(output_dir)
        
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")


if __name__ == "__main__":
    main()
