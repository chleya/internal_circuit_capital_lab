@echo off
cd /d F:\internal_circuit_capital_lab\IC-4-M0
echo ========================================================
echo Phase 1: n=5 only (rep 3-4)
echo ========================================================
python -u src/run_branch_a2_degradation_curve.py --mode primary --resume --n-list 5 2>&1
echo.
echo EXIT CODE: %ERRORLEVEL%
echo.
echo Phase 1 DONE.