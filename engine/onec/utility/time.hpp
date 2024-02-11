#pragma once

#include <ctime>
#include <string>

namespace onec
{

std::string formatDateTime(const std::tm& dateTime, std::string_view format = "%Y-%m-%d %H-%M-%S");
std::tm getDateTime();

}
