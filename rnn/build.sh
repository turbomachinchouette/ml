#!/bin/sh
refer -p ../refer.db rnn_report.ms | pic | groff -e -ms -Tdvi > report.dvi
dvipdf report.dvi


