import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from war_processor import WARProcessor
from value_model import PlayerValueModel


# Fixtures
@pytest.fixture
def sample_war_data():
    """Create sample WAR data for testing"""
    data = {
        'year': [2020, 2020, 2021, 2021, 2022, 2022] * 3,
        'player_id': ['p1', 'p2'] * 9,
        'name': ['Player 1', 'Player 2'] * 9,
        'age': [25, 28, 26, 29, 27, 30] * 3,
        'position': ['SS', '1B'] * 9,
        'war': [5.2, 3.8, 6.1, 4.2, 5.8, 3.5] * 3,
        'games_played': [150, 140, 155, 145, 152, 142] * 3
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_contract_data():
    """Create sample contract data"""
    arb_data = {
        'player_id': ['p1', 'p1', 'p2'],
        'year': [2020, 2021, 2020],
        'salary': [2_000_000, 3_500_000, 5_000_000],
        'total_value': [2_000_000, 3_500_000, 5_000_000],
        'contract_length': [1, 1, 1]
    }
    
    fa_data = {
        'player_id': ['p2', 'p3'],
        'year': [2021, 2022],
        'salary': [8_000_000, 12_000_000],
        'total_value': [24_000_000, 36_000_000],
        'contract_length': [3, 3]
    }
    
    return pd.DataFrame(arb_data), pd.DataFrame(fa_data)


class TestWARProcessor:
    """Tests for WARProcessor class"""
    
    def test_initialization(self, sample_war_data):
        """Test processor initialization"""
        processor = WARProcessor(sample_war_data)
        assert len(processor.war_data) == len(sample_war_data)
        assert 'war' in processor.war_data.columns
    
    def test_calculate_rookie_years(self, sample_war_data):
        """Test rookie year calculation"""
        processor = WARProcessor(sample_war_data)
        rookie_years = processor.calculate_rookie_years()
        
        assert 'player_id' in rookie_years.columns
        assert 'rookie_year' in rookie_years.columns
        assert len(rookie_years) == sample_war_data['player_id'].nunique()
    
    def test_add_career_year(self, sample_war_data):
        """Test career year calculation"""
        processor = WARProcessor(sample_war_data)
        df = processor.add_career_year()
        
        assert 'career_year' in df.columns
        assert df['career_year'].min() >= 0
        assert not df['career_year'].isna().any()
    
    def test_aggregate_by_age(self, sample_war_data):
        """Test age aggregation"""
        processor = WARProcessor(sample_war_data)
        age_agg = processor.aggregate_by_age()
        
        assert not age_agg.empty
        assert 'age' in age_agg.columns
        assert 'sum_war' in age_agg.columns
        assert 'mean_war' in age_agg.columns
    
    def test_aggregate_by_position(self, sample_war_data):
        """Test position aggregation"""
        processor = WARProcessor(sample_war_data)
        pos_agg = processor.aggregate_by_position()
        
        if not pos_agg.empty:
            assert 'position' in pos_agg.columns
            assert 'sum_war' in pos_agg.columns
    
    def test_aggregate_by_career_year(self, sample_war_data):
        """Test career year aggregation"""
        processor = WARProcessor(sample_war_data)
        career_agg = processor.aggregate_by_career_year()
        
        assert not career_agg.empty
        assert 'career_year' in career_agg.columns
        assert 'sum_war' in career_agg.columns
    
    def test_replacement_level(self, sample_war_data):
        """Test replacement level calculation"""
        processor = WARProcessor(sample_war_data)
        replacement = processor.calculate_replacement_level(percentile=20)
        
        assert isinstance(replacement, (int, float))
        assert replacement < sample_war_data['war'].mean()
    
    def test_war_above_replacement(self, sample_war_data):
        """Test WAR above replacement calculation"""
        processor = WARProcessor(sample_war_data)
        df = processor.calculate_war_above_replacement()
        
        assert 'warp' in df.columns
        assert (df['warp'] >= 0).all()
    
    def test_get_player_war_history(self, sample_war_data):
        """Test player WAR history retrieval"""
        processor = WARProcessor(sample_war_data)
        history = processor.get_player_war_history('p1')
        
        assert not history.empty
        assert (history['player_id'] == 'p1').all()
        assert 'cumulative_war' in history.columns
    
    def test_top_players_by_war(self, sample_war_data):
        """Test top players retrieval"""
        processor = WARProcessor(sample_war_data)
        top_players = processor.get_top_players_by_war(year=2020, top_n=5)
        
        assert len(top_players) <= 5
        assert top_players['war'].is_monotonic_decreasing


class TestPlayerValueModel:
    """Tests for PlayerValueModel class"""
    
    def test_initialization(self, sample_war_data, sample_contract_data):
        """Test model initialization"""
        arb_contracts, fa_contracts = sample_contract_data
        model = PlayerValueModel(sample_war_data, arb_contracts, fa_contracts)
        
        assert len(model.war_data) > 0
        assert len(model.arb_contracts) > 0
        assert len(model.fa_contracts) > 0
    
    def test_calculate_dollars_per_war(self, sample_war_data, sample_contract_data):
        """Test $/WAR calculation"""
        arb_contracts, fa_contracts = sample_contract_data
        model = PlayerValueModel(sample_war_data, arb_contracts, fa_contracts)
        
        rates = model.calculate_dollars_per_war(year=2020)
        
        assert isinstance(rates, dict)
        assert 'arbitration' in rates
        assert 'free_agency' in rates
    
    def test_historical_market_rates(self, sample_war_data, sample_contract_data):
        """Test historical rate calculation"""
        arb_contracts, fa_contracts = sample_contract_data
        model = PlayerValueModel(sample_war_data, arb_contracts, fa_contracts)
        
        rates = model.calculate_historical_market_rates(start_year=2020)
        
        assert not rates.empty
        assert 'year' in rates.columns
        assert 'arb_dollars_per_war' in rates.columns
    
    def test_player_surplus_value(self, sample_war_data, sample_contract_data):
        """Test surplus value calculation"""
        arb_contracts, fa_contracts = sample_contract_data
        model = PlayerValueModel(sample_war_data, arb_contracts, fa_contracts)
        
        market_rates = {'arbitration': 4_000_000, 'free_agency': 8_000_000}
        surplus = model.calculate_player_surplus_value('p1', 2020, market_rates)
        
        if surplus:
            assert 'surplus_value' in surplus
            assert 'market_value' in surplus
            assert 'actual_cost' in surplus
    
    def test_trade_value_index(self, sample_war_data, sample_contract_data):
        """Test trade value index calculation"""
        arb_contracts, fa_contracts = sample_contract_data
        model = PlayerValueModel(sample_war_data, arb_contracts, fa_contracts)
        
        trade_value = model.calculate_trade_value_index('p1')
        
        if trade_value:
            assert 'trade_value_index' in trade_value
            assert 0 <= trade_value['trade_value_index'] <= 100
            assert 'age_factor' in trade_value
    
    def test_compare_contract_types(self, sample_war_data, sample_contract_data):
        """Test contract type comparison"""
        arb_contracts, fa_contracts = sample_contract_data
        model = PlayerValueModel(sample_war_data, arb_contracts, fa_contracts)
        
        comparison = model.compare_contract_types()
        
        assert not comparison.empty
        assert 'arbitration' in comparison.columns
        assert 'free_agency' in comparison.columns


class TestDataValidation:
    """Tests for data validation and edge cases"""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrames"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            processor = WARProcessor(empty_df)
    
    def test_missing_columns(self):
        """Test handling of missing required columns"""
        df = pd.DataFrame({'year': [2020], 'player_id': ['p1']})
        
        with pytest.raises(ValueError):
            processor = WARProcessor(df)
    
    def test_negative_war_values(self, sample_war_data):
        """Test handling of negative WAR values"""
        sample_war_data.loc[0, 'war'] = -1.5
        processor = WARProcessor(sample_war_data)
        
        # Should still process without error
        age_agg = processor.aggregate_by_age()
        assert not age_agg.empty
    
    def test_missing_age_data(self, sample_war_data):
        """Test handling of missing age data"""
        sample_war_data.loc[0:2, 'age'] = np.nan
        processor = WARProcessor(sample_war_data)
        
        age_agg = processor.aggregate_by_age()
        # Should skip rows with missing age
        assert len(age_agg) < len(sample_war_data)


class TestIntegration:
    """Integration tests"""
    
    def test_full_pipeline(self, sample_war_data, sample_contract_data, tmp_path):
        """Test complete analysis pipeline"""
        arb_contracts, fa_contracts = sample_contract_data
        
        # Process WAR data
        processor = WARProcessor(sample_war_data)
        processor.save_aggregations(tmp_path)
        
        # Calculate values
        model = PlayerValueModel(sample_war_data, arb_contracts, fa_contracts)
        model.save_value_metrics(tmp_path)
        
        # Verify outputs
        assert (tmp_path / "war_by_age.csv").exists()
        assert (tmp_path / "market_rates_history.csv").exists()


def run_tests():
    """Run all tests"""
    pytest.main([__file__, '-v', '--cov=.', '--cov-report=term-missing'])


if __name__ == "__main__":
    run_tests()
