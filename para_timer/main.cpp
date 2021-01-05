#include <chrono>
#include <climits>
#include <iostream>
#include <istream>
#include <iomanip>
#include <numeric>

#include "../para/include/gmhc.hpp"
#include "option.hpp"
#include "reader.hpp"

std::string enum_to_string(clustering::subthreshold_handling_kind kind)
{
    switch (kind)
    {
        case clustering::subthreshold_handling_kind::MAHAL:
            return "M";
        case clustering::subthreshold_handling_kind::EUCLID:
            return "E";
        case clustering::subthreshold_handling_kind::MAHAL0:
            return "M0";
        case clustering::subthreshold_handling_kind::EUCLID_MAHAL:
            return "EM";
        default:
            return "UNDEF";
    }
}

void print_time(std::vector<std::chrono::duration<double>>& time, int reps, clustering::subthreshold_handling_kind kind)
{
    std::cout << enum_to_string(kind) << "\t";

    auto sum = std::accumulate(time.begin(), time.end(), std::chrono::duration<double>::zero());
    auto mean = sum / reps;
    std::cout << mean.count() << "\t";

    double std_dev = 0;
    for (auto t : time)
        std_dev += (t - mean).count() * (t - mean).count();
    std_dev /= reps;
    std_dev = std::sqrt(std_dev);
    std::cout << std_dev << std::endl;
    time.clear();
}

int measure(const float* data, clustering::csize_t count, clustering::csize_t dim, int repetitions)
{
    using namespace clustering;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "kind\ttime    \tstd.dev" << std::endl; 

    auto threshold = (csize_t)(count * 0.5);

    // dry run
    {
        gmhc gmhclust;
        if (!gmhclust.initialize(data, count, dim, threshold, subthreshold_handling_kind::MAHAL))
            return 1;
        gmhclust.run();
        gmhclust.free();
    }

    gmhc gmhclust;

    std::vector<std::chrono::duration<double>> run_time;
    std::vector<double> processed_time;
    size_t enum_size = 4;

    for (size_t i = 0; i < enum_size; i++)
    {
        auto kind = (subthreshold_handling_kind)i;

        for (size_t j = 0; j < repetitions; j++)
        {
            auto start = std::chrono::system_clock::now();

            if (!gmhclust.initialize(data, count, dim, threshold, kind))
                return 1;
            gmhclust.run();
            gmhclust.free();

            auto end = std::chrono::system_clock::now();

            run_time.push_back(end - start);
        }
        print_time(run_time, repetitions, kind);
    }

    return 0;
}

int main(int argc, char** argv)
{
    using namespace mbas;
    using namespace clustering;

    command cmd;

    cmd.add_option("d,dataset", "Path to dataset file.", false)
        .add_parameter<std::string>(value_type<std::string>(), "D_PATH");
    cmd.add_option("n", "Test repetition.", false)
        .add_parameter<int>(value_type<int>(), "REPS")
        .set_constraint([](int value) { return value >= 1; });

    auto parsed = cmd.parse(argc, argv);

    if (!parsed.parse_ok())
    {
        std::cerr << cmd.help();
        return 1;
    }

    auto dataset = parsed["dataset"]->get_value<std::string>();
    auto data = reader::read_data_from_binary_file<float, double>(dataset);
    auto reps = parsed["n"]->get_value<int>();

    return measure(data.data.data(), (csize_t)data.points, (csize_t)data.dim, reps);
}
