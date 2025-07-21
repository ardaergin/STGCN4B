import holidays
from datetime import datetime
from typing import List, Tuple, Dict

import logging; logger = logging.getLogger(__name__)


class WorkHourClassifier:
    """
    Class to classify time periods as work hours or non-work hours,
    accounting for holidays, weekends, and standard office hours.
    """
    
    def __init__(
        self, 
        country_code="NL", 
        start_year=None, 
        end_year=None,
        workhour_start=8,
        workhour_end=18
    ):
        """
        Initialize the classifier with holiday data for the specified country.
        
        Args:
            country_code: Country code for holidays (default: NL for Netherlands)
            start_year: Starting year for holiday data (default: current year)
            end_year: Ending year for holiday data (default: current year)
        """
        # Workhour parameters
        self.workhour_start = workhour_start
        self.workhour_end = workhour_end
        
        # Initialize years for holiday data
        if start_year is None or end_year is None:
            current_year = datetime.now().year
            start_year = start_year or current_year
            end_year = end_year or current_year
        
        # Initialize holiday calendar
        self.holidays_calendar = self._initialize_holidays(country_code, start_year, end_year)
        
        logger.info(f"Initialized WorkHourClassifier with {country_code} holidays for {start_year}-{end_year}")
    
    def _initialize_holidays(self, country_code: str, start_year: int, end_year: int):
        """
        Initialize holiday calendar for the specified country and years.
        
        Args:
            country_code: Country code (e.g., "NL" for Netherlands)
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            holidays.HolidayBase: Holiday calendar object
        """
        # Get the appropriate country calendar from the holidays package
        if country_code == "NL":
            calendar = holidays.NL(years=range(start_year, end_year + 1))
        else:
            # Add support for other countries as needed
            logger.warning(f"Country code {country_code} not specifically supported. Using generic holidays.")
            calendar = holidays.country_holidays(country_code, years=range(start_year, end_year + 1))
        
        return calendar
    
    def _is_work_hour(self, timestamp: datetime) -> bool:
        """
        Determine if a specific timestamp is during work hours.
        
        Args:
            timestamp: The datetime to check
            
        Returns:
            bool: True if it's a work hour, False otherwise
        """
        # Convert datetime to date for holiday checking
        current_date = timestamp.date()
        
        # First check if it's a holiday
        if current_date in self.holidays_calendar:
            return False
        
        # If not a holiday, check if it's a weekday (0-4 is Monday-Friday)
        day_of_week = timestamp.weekday()
        if day_of_week >= 5:  # Weekend
            return False
        
        # Check if it's between standard office hours (9:00-17:00)
        hour_of_day = timestamp.hour
        return self.workhour_start <= hour_of_day < self.workhour_end
    
    def classify_time_buckets(
            self, 
            time_buckets: List[Tuple[datetime, datetime]]
    ) -> Dict[int, int]:
        """
        Classify a list of time buckets as work hour (1) or non-work hour (0).
        
        Args:
            time_buckets: List of (start_time, end_time) tuples
        
        Returns:
            {bucket_idx: label}: Binary labels (1 for work hour, 0 for non-work hour)
        """
        labels = {}
        holiday_hours = work_hours = non_work_hours = 0
        
        for idx, (start, _) in enumerate(time_buckets):
            if self._is_work_hour(start):
                labels[idx] = 1
                work_hours += 1
            else:
                labels[idx] = 0
                non_work_hours += 1
                if start.date() in self.holidays_calendar:
                    holiday_hours += 1
        
        logger.info(
            f"Classified {len(labels)} buckets: "
            f"{work_hours} work, {non_work_hours} non‑work "
            f"({holiday_hours} on holidays)"
        )
        baseline = max(work_hours, non_work_hours) / len(labels) * 100
        logger.info(f"Baseline accuracy: {baseline:.2f}%")
        return labels