#!/usr/bin/env python
import sys
import os
from collections import OrderedDict

class Logger:
  def __init__(self, file_name):
    self.log_data = OrderedDict()
    self.file = file_name
    self.header_written = False

  def update(self, key, val):
    self.log_data[key] = val

  def write(self):
    t = ""
    h = ""
    for key, value in self.log_data.items():
      if not self.header_written:
        h += str(key) + ','
      t += str(value) + ','

    if not self.header_written:
      with open(self.file, "w") as fn:
        fn.write(h+'\n')
    with open(self.file, "a") as fn:
      fn.write(t+'\n')