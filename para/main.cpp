#include <climits>
#include <iostream>
#include <istream>

#include "gmhc.hpp"
#include "reader.hpp"

std::vector<clustering::asgn_t> create_apriori_assigns(const char* file_name, size_t count)
{
    std::ifstream fs(file_name);
    std::vector<clustering::asgn_t> ret;

    if (!fs.is_open())
    {
        std::cerr << "file could not open" << std::endl;
        return {};
    }

    for (size_t i = 0; i < count; ++i)
    {
        clustering::asgn_t tmp;
        fs >> tmp;
        ret.push_back(tmp);

        if (fs.fail() || fs.bad())
        {
            std::cerr << "issue when reading data";
            return {};
        }
    }
    return ret;
}

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 5)
    {
        std::cout << "bad input" << std::endl
                  << "usage: mhclust file_name maha_threshold (MAHAL|MAHAL0|EUCLID|EUCLID_MAHAL) [apriori_file]"
                  << std::endl;
        return 1;
    }

    auto data = clustering::reader::read_data_from_binary_file<float>(argv[1]);

    std::vector<clustering::asgn_t> apriori_assignments;
    clustering::asgn_t* apr_asgn = nullptr;

    char* end;
    auto thresh = std::strtoll(argv[2], &end, 10);

    if (thresh >= (long long)ULONG_MAX || thresh < 0 || *end != '\0')
    {
        std::cerr << "bad threshold" << std::endl;
        return 1;
    }

    clustering::subthreshold_handling_kind kind;
    std::string input_kind(argv[3]);
    if (input_kind == "MAHAL")
        kind = clustering::subthreshold_handling_kind::MAHAL;
    else if (input_kind == "MAHAL0")
        kind = clustering::subthreshold_handling_kind::MAHAL0;
    else if (input_kind == "EUCLID")
        kind = clustering::subthreshold_handling_kind::EUCLID;
    else if (input_kind == "EUCLID_MAHAL")
        kind = clustering::subthreshold_handling_kind::EUCLID_MAHAL;
    else
    {
        std::cerr << "bad subthreshold kind, MAHAL used";
        kind = clustering::subthreshold_handling_kind::MAHAL;
    }

    if (argc == 5)
    {
        apriori_assignments = create_apriori_assigns(argv[4], data.points);
        if (apriori_assignments.empty())
            return 1;
        apr_asgn = apriori_assignments.data();
    }

    clustering::gmhc gmhclust;

    bool init = gmhclust.initialize(data.data.data(),
        (clustering::csize_t)data.points,
        (clustering::csize_t)data.dim,
        (clustering::csize_t)thresh,
        kind,
        apr_asgn);

    if (!init)
        return 1;

    auto res = gmhclust.run();

    for (auto& e : res)
        std::cout << e.first.first << " " << e.first.second << " " << e.second << std::endl;

    gmhclust.free();

    return 0;
}
