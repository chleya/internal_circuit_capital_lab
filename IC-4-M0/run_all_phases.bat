@echo off
setlocal enabledelayedexpansion
cd /d F:\internal_circuit_capital_lab\IC-4-M0

set LOG=results_branch_a2\runner_log.txt
set CSV=results_branch_a2\degradation_curve.csv

echo ================================================================ >> %LOG%
echo Runner started at %date% %time% >> %LOG%
echo ================================================================ >> %LOG%

REM Phase 1: n=5 rep=4 only
echo [%time%] Starting Phase 1: n=5 rep=4 >> %LOG%
python -u src/run_branch_a2_degradation_curve.py --mode primary --resume --n-list 5 >> %LOG% 2>&1
set EXIT=%ERRORLEVEL%
echo [%time%] Phase 1 DONE exit=%EXIT% >> %LOG%

REM Clean CSV duplicates
echo [%time%] Deduplicating CSV... >> %LOG%
python -u src/clean_csv.py >> %LOG% 2>&1
echo [%time%] CSV cleaned >> %LOG%

REM Phase 2: n=3 all 5 reps
echo [%time%] Starting Phase 2: n=3 >> %LOG%
python -u src/run_branch_a2_degradation_curve.py --mode primary --resume --n-list 3 >> %LOG% 2>&1
set EXIT=%ERRORLEVEL%
echo [%time%] Phase 2 DONE exit=%EXIT% >> %LOG%

REM Clean CSV duplicates
echo [%time%] Deduplicating CSV... >> %LOG%
python -u src/clean_csv.py >> %LOG% 2>&1

REM Phase 3: n=2 all 5 reps
echo [%time%] Starting Phase 3: n=2 >> %LOG%
python -u src/run_branch_a2_degradation_curve.py --mode primary --resume --n-list 2 >> %LOG% 2>&1
set EXIT=%ERRORLEVEL%
echo [%time%] Phase 3 DONE exit=%EXIT% >> %LOG%

REM Clean CSV duplicates
echo [%time%] Deduplicating CSV... >> %LOG%
python -u src/clean_csv.py >> %LOG% 2>&1

REM Phase 4: harder eval + report
echo [%time%] Starting Phase 4: harder eval >> %LOG%
python -u src/run_branch_a2_degradation_curve.py --mode harder >> %LOG% 2>&1
set EXIT=%ERRORLEVEL%
echo [%time%] Phase 4 DONE exit=%EXIT% >> %LOG%

echo [%time%] ALL PHASES COMPLETE >> %LOG%
echo ================================================================ >> %LOG%