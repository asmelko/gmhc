#ifndef READER_H
#define READER_H

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "clustering.hpp"

namespace clustering {

// reads input data
struct reader
{
    template<typename T>
    struct data_t
    {
        size_t dim;
        size_t points;
        std::vector<T> data;
    };

    template<typename Out_T, typename Read_T = Out_T>
    static data_t<Out_T> read_data_from_binary_file(const std::string& file)
    {
        std::ifstream fs(file, std::ios::in | std::ios::binary);

        if (!fs.is_open())
        {
            std::cerr << "file could not open";
            return { 0, 0, std::vector<Out_T> {} };
        }

        return read_bin_data<Out_T, Read_T>(fs);
    }

    template<typename T>
    static data_t<T> read_data_from_file(const std::string& file)
    {
        std::ifstream fs(file);

        if (!fs.is_open())
        {
            std::cerr << "file could not open";
            return { 0, 0, std::vector<T> {} };
        }

        return read_data<T>(fs);
    }

    template<typename T>
    static data_t<T> read_data_from_string(const std::string& data)
    {
        std::stringstream ss(data, std::ios_base::in);
        return read_data<T>(ss);
    }

    static std::vector<asgn_t> read_assignments_from_file(const std::string& file)
    {
        std::ifstream fs(file);

        if (!fs.is_open())
        {
            std::cerr << "file could not open";
            return {};
        }

        std::vector<asgn_t> ret;

        while (true)
        {
            int asgn;
            fs >> asgn;

            if (fs.eof())
                break;

            if (fs.fail() || fs.bad())
            {
                std::cerr << "issue when reading data";
                return {};
            }

            ret.push_back(asgn);
        }

        return ret;
    }

private:
    template<typename T>
    static data_t<T> read_data(std::istream& stream)
    {
        size_t dim, points;

        stream >> dim >> points;

        std::vector<T> ret;
        ret.resize(dim * points);

        size_t idx = 0;

        for (size_t i = 0; i < points * dim; ++i)
        {
            stream >> ret[idx++];
            if (stream.fail() || stream.bad())
            {
                std::cerr << "issue when reading data";
                return { 0, 0, std::vector<T> {} };
            }
        }

        return { dim, points, std::move(ret) };
    }

    template<typename Out_T, typename Read_T = Out_T>
    static data_t<Out_T> read_bin_data(std::istream& stream)
    {
        std::uint64_t dim, points;
        Read_T point;

        stream.read((char*)&dim, sizeof(dim));
        stream.read((char*)&points, sizeof(points));

        std::vector<Out_T> ret;
        ret.reserve(dim * points);

        for (size_t i = 0; i < points * dim; ++i)
        {
            stream.read((char*)&point, sizeof(point));
            ret.push_back(static_cast<Out_T>(point));

            if (stream.fail() || stream.bad())
            {
                std::cerr << "issue when reading data";
                return { 0, 0, std::vector<Out_T> {} };
            }
        }

        return { dim, points, std::move(ret) };
    }
};

} // namespace clustering

#endif