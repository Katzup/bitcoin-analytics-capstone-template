#!/bin/bash
cd /Users/bobkatz/Visual_Trading_System/final_report

echo "=== Pass 1: vts_final_report.tex ===" > /tmp/latex_run.log
pdflatex -interaction=nonstopmode vts_final_report.tex >> /tmp/latex_run.log 2>&1

echo "=== Pass 2: vts_final_report.tex ===" >> /tmp/latex_run.log
pdflatex -interaction=nonstopmode vts_final_report.tex >> /tmp/latex_run.log 2>&1

echo "=== Pass 1: vts_final_report_gt.tex ===" >> /tmp/latex_run.log
pdflatex -interaction=nonstopmode vts_final_report_gt.tex >> /tmp/latex_run.log 2>&1

echo "=== Pass 2: vts_final_report_gt.tex ===" >> /tmp/latex_run.log
pdflatex -interaction=nonstopmode vts_final_report_gt.tex >> /tmp/latex_run.log 2>&1

echo "DONE" >> /tmp/latex_run.log
