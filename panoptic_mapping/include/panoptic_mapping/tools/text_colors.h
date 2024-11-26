/*
 * @Author: thuaj@connect.ust.hk
 * @Date: 2023-03-30 16:42:05
 * @LastEditTime: 2023-07-12 16:59:38
 * @Description: std::cout text coloring
 * Copyright (c) 2023 by thuaj@connect.ust.hk, All Rights Reserved.
 */
#ifndef PANOPTIC_MAPPING_TOOLS_TEXT_COLORS_H_
#define PANOPTIC_MAPPING_TOOLS_TEXT_COLORS_H_

#include <iostream>

// color text std::cout
typedef std::ostream& (*BlueText)(std::ostream&);
typedef std::ostream& (*RedText)(std::ostream&);
typedef std::ostream& (*GreenText)(std::ostream&);
typedef std::ostream& (*EndColor)(std::ostream&);
typedef std::ostream& (*OrangeText)(std::ostream&);
typedef std::ostream& (*PurpleText)(std::ostream&);
typedef std::ostream& (*PinkText)(std::ostream&);
// hight background
typedef std::ostream& (*YellowHighlight)(std::ostream&);

typedef std::ostream& (*CyanText)(std::ostream&);

static RedText redText = [](std::ostream& os) -> std::ostream& {
  return os << "\033[0;31m";
};

static GreenText greenText = [](std::ostream& os) -> std::ostream& {
  return os << "\033[0;32m";
};

static OrangeText orangeText = [](std::ostream& os) -> std::ostream& {
  return os << "\033[0;33m";
};

static BlueText blueText = [](std::ostream& os) -> std::ostream& {
  return os << "\033[0;34m";
};

static PurpleText purpleText = [](std::ostream& os) -> std::ostream& {
  return os << "\033[0;35m";
};

static CyanText cyanText = [](std::ostream& os) -> std::ostream& {
  return os << "\033[0;36m";
};

static PinkText pinkText = [](std::ostream& os) -> std::ostream& {
  return os << "\033[0;95m";
};

static YellowHighlight yellowHighlight = [](std::ostream& os) -> std::ostream& {
  return os << "\033[43m";
};

static EndColor endColor = [](std::ostream& os) -> std::ostream& {
  return os << "\033[0m";
};

#endif  // PANOPTIC_MAPPING_TOOLS_TEXT_COLORS_H_
