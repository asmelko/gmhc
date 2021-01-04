#include <chrono>
#include <climits>
#include <iostream>
#include <istream>
#include <numeric>

#include "../para/include/gmhc.hpp"
#include "reader.hpp"


std::string enum_to_string(clustering::subthreshold_handling_kind kind)
{
    switch (kind)
    {
        case clustering::subthreshold_handling_kind::MAHAL:
            return "MAHAL";
        case clustering::subthreshold_handling_kind::EUCLID:
            return "EUCLID";
        case clustering::subthreshold_handling_kind::MAHAL0:
            return "MAHAL0";
        case clustering::subthreshold_handling_kind::EUCLID_MAHAL:
            return "EUCLID_MAHAL";
        default:
            return "UNK";
    }
}

void print_time(const std::vector<std::chrono::duration<double>>& time, long reps, std::vector<double>& ret)
{
    auto sum = std::accumulate(time.begin(), time.end(), std::chrono::duration<double>::zero());
    auto mean = sum / reps;
    ret.push_back(mean.count());
    std::cerr << mean.count() << std::endl;

    double std_dev = 0;
    for (auto t : time)
        std_dev += (t - mean).count() * (t - mean).count();
    std_dev /= reps;
    std_dev = std::sqrt(std_dev);
    ret.push_back(std_dev);
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cout << "bad input" << std::endl << "usage: para_timer repetitions file_name" << std::endl;
        return 1;
    }

    char* r_end;
    auto reps = std::strtol(argv[1], &r_end, 10);

    if (reps < 0 || *r_end != '\0')
    {
        std::cerr << "bad repetitions number" << std::endl;
        return 1;
    }

    auto data = clustering::reader::read_data_from_binary_file<float, double>(argv[2]);

    auto thresh = (clustering::csize_t)(data.points * 0.5);
    // dry run
    clustering::gmhc gmhclust;
    if (!gmhclust.initialize(data.data.data(),
            (clustering::csize_t)data.points,
            (clustering::csize_t)data.dim,
            thresh,
            clustering::subthreshold_handling_kind::MAHAL))
        return 1;
    gmhclust.run();
    gmhclust.free();

    std::vector<std::chrono::duration<double>> run_time;
    std::vector<double> processed_time;
    size_t enum_size = 4;

    for (size_t i = 0; i < enum_size; i++)
    {
        auto sub_kind = (clustering::subthreshold_handling_kind)i;

        for (size_t j = 0; j < reps; j++)
        {
            auto start = std::chrono::system_clock::now();

            if (!gmhclust.initialize(data.data.data(),
                    (clustering::csize_t)data.points,
                    (clustering::csize_t)data.dim,
                    thresh,
                    sub_kind))
                return 1;
            gmhclust.run();
            gmhclust.free();

            auto end = std::chrono::system_clock::now();

            run_time.push_back(end - start);
        }
        print_time(run_time, reps, processed_time);
        run_time.clear();
    }

    std::cout << "(M,E,M0,EM): (";
    for (size_t i = 0; i < enum_size; i++) 
    {
        std::cout << processed_time[i * 2];
        if (i != enum_size - 1)
            std::cout << ",";
        else
            std::cout << ") avg.time, (";
    }
    for (size_t i = 0; i < enum_size; i++)
    {
        std::cout << processed_time[i * 2 + 1];
        if (i != enum_size - 1)
            std::cout << ",";
        else
            std::cout << ") std.dev." << std::endl;
    }

    return 0;
}
