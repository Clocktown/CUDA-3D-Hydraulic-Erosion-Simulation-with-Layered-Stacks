#include "time.hpp"
#include "../config/config.hpp"
#include <errno.h>
#include <ctime>
#include <string>
#include <vector>

namespace onec
{

std::string formatDateTime(const std::tm& dateTime, const std::string_view format)
{
	if (format.empty())
	{
		return std::string{};
	}

	std::vector<char> buffer(32);
	std::size_t count;

	do
	{
		count = std::strftime(buffer.data(), buffer.size(), format.data(), &dateTime);

		if (count == 0)
		{
			buffer.resize(2 * buffer.size());
		}
		else
		{
			break;
		}
	}
	while (true);

	return std::string{ buffer.data(), count };
}

std::tm getDateTime()
{
	const std::time_t time{ std::time(nullptr) };
	std::tm dateTime;

	const errno_t error{ localtime_s(&dateTime, &time) };

	ONEC_ASSERT(error == errno_t{}, "Failed to get date time");

	return dateTime;
}

}
