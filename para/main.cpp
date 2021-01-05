#include <fstream>
#include <iostream>

#include "gmhc.hpp"
#include "option.hpp"
#include "reader.hpp"

int main(int argc, char** argv)
{
    using namespace mbas;
    using namespace clustering;

    command cmd;

    cmd.add_option("-d,dataset", "Path to dataset file.", false)
        .add_parameter<std::string>(value_type<std::string>(), "D_PATH");
    cmd.add_option("-a,apriori", "Path to apriori assignments file.", true)
        .add_parameter<std::string>(value_type<std::string>(), "A_PATH");
    cmd.add_option("-t,threshold", "Mahalanobis threshold. Allowed values are [0,1].", false)
        .add_parameter<float>(value_type<float>(), "THRESHOLD")
        .set_constraint([](float value) { return value >= 0 && value <= 1; });
    cmd.add_option("-k,kind", "Subthreshold kind. Allowed values are (MAHAL|MAHAL0|EUCLID|EUCLID_MAHAL).", true)
        .add_parameter<std::string>(value_type<std::string>(), "KIND")
        .set_constraint([](const std::string& value) {
            return value == "MAHAL" || value == "MAHAL0" || value == "EUCLID" || value == "EUCLID_MAHAL";
        });

    auto parsed = cmd.parse(argc, argv);

    if (!parsed.parse_ok())
    {
        std::cerr << cmd.help();
        return 1;
    }

    auto dataset = parsed["dataset"]->get_value<std::string>();
    auto data = reader::read_data_from_binary_file<float, double>(dataset);

    std::vector<asgn_t> apriori_assignments;
    asgn_t* apr_asgn = nullptr;
    if (parsed["apriori"])
    {
        auto apriori_file = parsed["apriori"]->get_value<std::string>();
        apriori_assignments = reader::read_assignments_from_file(apriori_file);
        if (apriori_assignments.size() != data.points)
        {
            std::cerr << "bad apriori assignment file";
            return 1;
        }
        apr_asgn = apriori_assignments.data();
    }

    auto kind = subthreshold_handling_kind::MAHAL;

    if (parsed["kind"])
    {
        auto input_kind = parsed["kind"]->get_value<std::string>();
        if (input_kind == "MAHAL")
            kind = subthreshold_handling_kind::MAHAL;
        else if (input_kind == "MAHAL0")
            kind = subthreshold_handling_kind::MAHAL0;
        else if (input_kind == "EUCLID")
            kind = subthreshold_handling_kind::EUCLID;
        else if (input_kind == "EUCLID_MAHAL")
            kind = subthreshold_handling_kind::EUCLID_MAHAL;
    }

    auto threshold = parsed["threshold"]->get_value<float>();

    gmhc gmhclust;
    auto actual_thresholh = (csize_t)((float)data.points * threshold);

    bool init = gmhclust.initialize(
        data.data.data(), (csize_t)data.points, (csize_t)data.dim, actual_thresholh, kind, apr_asgn);

    if (!init)
        return 1;

    auto res = gmhclust.run();

    for (auto& e : res)
        std::cout << e.first.first << " " << e.first.second << " " << e.second << std::endl;

    gmhclust.free();

    return 0;
}
