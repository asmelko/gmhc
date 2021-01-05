#ifndef OPTION_HPP
#define OPTION_HPP

#include <algorithm>
#include <deque>
#include <functional>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mbas {

namespace internal {
class parameter_base_accessor;
}

// represents general parameter of an option
class parsed_param_base
{
public:
    parsed_param_base(std::string name, bool parse_ok, bool constr_ok)
        : name_(std::move(name))
        , p_ok_(parse_ok)
        , constr_ok_(constr_ok)
    {}

    const std::string& name() const { return name_; }

    // gets the parsed value
    template<typename T>
    const T& get_value() const;

    // false iff the argument could not be parsed into the specified type
    bool parse_ok() const { return p_ok_; }
    // false iff the value does not conform to the specified constraint
    bool constraint_ok() const { return constr_ok_; }

    virtual ~parsed_param_base() {}

private:
    const std::string name_;
    const bool p_ok_;
    const bool constr_ok_;
};

template<typename T>
class parsed_param : public parsed_param_base
{
public:
    parsed_param(T parsed_value, std::string name, bool parse_ok, bool constr_ok)
        : parsed_param_base(std::move(name), parse_ok, constr_ok)
        , parsed_value(std::move(parsed_value))
    {}

    const T parsed_value;
};

template<typename T>
const T& parsed_param_base::get_value() const
{
    if (!p_ok_)
        throw std::runtime_error("get_value called, but no value was parsed.");
    auto conv = dynamic_cast<const parsed_param<T>*>(this);
    if (!conv)
        throw std::runtime_error("get_value used with wrong type.");
    return conv->parsed_value;
}


class parsed_option
{
public:
    parsed_option(std::string name,
        std::vector<std::unique_ptr<parsed_param_base>> params,
        bool parse_ok,
        bool constr_ok,
        bool count_ok,
        bool unknown)
        : name_(std::move(name))
        , params_(std::move(params))
        , p_ok_(parse_ok)
        , constr_ok_(constr_ok)
        , count_ok_(count_ok)
        , unknown_(unknown)
    {}

    const std::string& name() { return name_; }

    const std::vector<std::unique_ptr<parsed_param_base>>& params() const { return params_; }

    template<typename T>
    const T& get_value() const
    {
        return params_[0]->get_value<T>();
    }

    // false, iff at least one of parameters of option could not be parsed (its parse_ok is false)
    bool parse_ok() const { return p_ok_; }
    // false, iff at least one of parameters does not conform to constraint
    bool constraints_ok() const { return constr_ok_; }
    // false, iff there were more mandatory parameters specified than parsed, or less parameters were
    // parsed, than the number of mandatory and optional parameters
    bool count_ok() const { return count_ok_; }

    // true, iff this option was not defined, but was parsed
    bool unknown() const { return unknown_; }

    parsed_option(const parsed_option& rhs) = delete;
    parsed_option(parsed_option&& rhs) noexcept
        : name_(std::move(rhs.name_))
        , params_(std::move(rhs.params_))
        , p_ok_(rhs.p_ok_)
        , constr_ok_(rhs.constr_ok_)
        , count_ok_(rhs.count_ok_)
        , unknown_(rhs.unknown_)
    {}

private:
    std::string name_;
    std::vector<std::unique_ptr<parsed_param_base>> params_;
    const bool p_ok_;
    const bool constr_ok_;
    const bool count_ok_;
    const bool unknown_;
};

class parsed_args
{
    std::unordered_map<std::string, parsed_option*> options_mapping_;
    std::deque<parsed_option> options_;

    std::vector<std::string> plain_arguments_;
    bool p_ok_;

public:
    parsed_args(std::unordered_map<std::string, parsed_option*> options_mapping,
        std::deque<parsed_option> options,
        std::vector<std::string> plain_arguments,
        bool parse_ok)
        : options_mapping_(std::move(options_mapping))
        , options_(std::move(options))
        , plain_arguments_(std::move(plain_arguments))
        , p_ok_(parse_ok)
    {}

    using const_iterator = decltype(options_)::const_iterator;

    // gets the list of all plain arguments
    const std::vector<std::string>& plain_args() const { return plain_arguments_; }

    // iterator to beginning and end of container with parsed options
    // naming convention of these methods allow the foreach semantics
    const_iterator begin() const { return options_.cbegin(); }
    const_iterator end() { return options_.cend(); }

    // returns parsed option with specified name, or null if it was not parsed
    const parsed_option* operator[](const std::string& option)
    {
        auto f = options_mapping_.find(option);
        return f == options_mapping_.end() ? nullptr : f->second;
    }
    // returns true, if option with specified name was parsed
    bool exists(const std::string& option) const { return options_mapping_.find(option) != options_mapping_.end(); }

    // returns, whether option was parsed and contains parameter with param_index.
    // fills parsed_value with the parsed parameter
    template<typename T>
    bool get_param(const std::string& option, size_t param_index, T& parsed_value);

    // false, iff there was any parsing error
    bool parse_ok() const { return p_ok_; }
};

// templated class that indicates the expected data type of argument.
// it may be easily specialised to parse custom types - all that is needed is to implement static parse function
template<typename T>
struct value_type;

template<>
struct value_type<int>
{
    static bool parse(std::string value, int& result)
    {
        try
        {
            result = std::stoi(value);
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
};

template<>
struct value_type<float>
{
    static bool parse(std::string value, float& result)
    {
        try
        {
            result = std::stof(value);
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
};

template<>
struct value_type<std::string>
{
    static bool parse(std::string value, std::string& result)
    {
        result = value;
        return true;
    }
};

template<>
struct value_type<bool>
{
    static bool parse(std::string value, bool& result)
    {
        std::transform(value.begin(), value.end(), value.begin(), [](auto& c) { return (char)std::toupper(c); });

        if (value == "TRUE" || value == "1")
        {
            result = true;
            return true;
        }
        if (value == "FALSE" || value == "0")
        {
            result = false;
            return true;
        }
        return false;
    }
};



// base class unifying parameter classes
class parameter_base
{
public:
    const std::string name;
    const bool mandatory;

    virtual ~parameter_base() {}

protected:
    parameter_base(const std::string name, bool mandatory)
        : name(name)
        , mandatory(mandatory)
    {}

private:
    friend internal::parameter_base_accessor;
    virtual std::unique_ptr<parsed_param_base> parse(std::string param) const = 0;
};

namespace internal {
class parameter_base_accessor
{
public:
    static std::unique_ptr<parsed_param_base> parse(const parameter_base& parameter, const std::string parsed);
};
} // namespace internal

using parameter_base_ptr = std::unique_ptr<parameter_base>;

// templated class representing parameter of an option
template<typename T>
class parameter : public parameter_base
{
public:
    // definition of constraint type for convenient use
    using constr_t = std::function<bool(T)>;

    // constructort of parameter with its NAME, whether it is MANDATORY and with its CONSTRAINT
    parameter(const std::string name, bool mandatory = true)
        : parameter_base(name, mandatory)
        , constraint_(nullptr)
    {}
    parameter(const std::string name, bool mandatory, constr_t constraint)
        : parameter_base(name, mandatory)
        , constraint_(constraint)
    {}

    // sets new constraint of the parameter
    void set_constraint(constr_t constraint) { constraint_ = constraint; }

private:
    virtual std::unique_ptr<parsed_param_base> parse(std::string param) const
    {
        T val;
        bool parse_ok = value_type<T>::parse(std::move(param), val);
        bool constr_ok = true;
        if (constraint_ && parse_ok)
            constr_ok = constraint_(val);

        return std::make_unique<parsed_param<T>>(std::move(val), name, parse_ok, constr_ok);
    }

    constr_t constraint_;
};

template<typename... T>
struct parameter_holder
{
    typedef std::tuple<parameter<T>...> type;
};

// class representing options to be parsed
class option
{
public:
    const std::vector<std::string> names;
    const std::string description;
    const bool optional;

    // option constructor with its NAMES, DESCRIPTION and OPTIONAL flag
    // in the first parameter synonyms are divided with comma where one character names are treated as short options and
    // two or more character names as long options
    option(std::string names, const std::string description, bool optional = true)
        : names(construct_names_(std::move(names)))
        , description(description)
        , optional(optional)
    {}

    // adds parameter to the option with its type (VALUE_TYPE<T>), NAME and MANDATORY flag
    // the first parameter does not have name because it serves as a deduction the type T and forces user to notice that
    // value_type<T> is needed to parse properly
    template<typename T>
    parameter<T>& add_parameter(value_type<T>, const std::string name, bool mandatory = true)
    {
        if (mandatory)
            ++mandatory_count_;
        params_.emplace_back(std::make_unique<parameter<T>>(std::move(name), mandatory));
        return *static_cast<parameter<T>*>(params_.back().get());
    }

    template<typename T>
    parameter<T>& add_parameter(parameter<T> param)
    {
        if (param.mandatory)
            ++mandatory_count_;
        params_.emplace_back(std::make_unique<parameter<T>>(std::move(param)));
        return *static_cast<parameter<T>*>(params_.back().get());
    }

    option(const option&) = delete;
    option(option&& opt) noexcept
        : names(opt.names)
        , description(opt.description)
        , optional(opt.optional)
        , params_(std::move(opt.params_))
        , mandatory_count_(opt.mandatory_count_)
    {}

    const std::vector<parameter_base_ptr>& parameters() const { return params_; }
    size_t mandatory_count() const { return mandatory_count_; }

private:
    std::vector<parameter_base_ptr> params_;

    size_t mandatory_count_ = 0;

    static std::vector<std::string> construct_names_(std::string&& names)
    {
        std::vector<std::string> res;

        size_t pos = 0;
        while ((pos = names.find(',')) != std::string::npos)
        {
            if (pos == 0)
                throw std::invalid_argument("Empty option alias specified");
            res.push_back(names.substr(0, pos));
            names.erase(0, pos + 1);
        }
        res.push_back(names.substr(0, pos));
        return res;
    }
};

namespace internal {
template<size_t I = 0, typename... Tp>
typename std::enable_if<I == sizeof...(Tp), void>::type add_param(std::tuple<Tp...>&, option&)
{}

template<size_t I = 0, typename... Tp>
    typename std::enable_if < I<sizeof...(Tp), void>::type add_param(std::tuple<Tp...>& t, option& opt)
{
    opt.add_parameter(std::get<I>(t));
    add_param<I + 1, Tp...>(t, opt);
}
} // namespace internal

// class representing the command itself holding info describing options
class command
{
    // class for quick and simple adding of not complex options
    class option_adder
    {
    public:
        option_adder(command& owner)
            : owner_(owner)
        {}

        // adds option with a parameter and its contraint
        template<typename T>
        option_adder& operator()(std::string names,
            const std::string description,
            value_type<T>,
            std::string param_name,
            typename ::mbas::parameter<T>::constr_t param_constraint)
        {
            owner_.add_option<T>(
                std::move(names), std::move(description), true, { parameter<T>(param_name, true, param_constraint) });
            return *this;
        }

        // adds option with a parameter
        template<typename T>
        option_adder& operator()(
            std::string names, const std::string description, value_type<T>, std::string param_name)
        {
            owner_.add_option<T>(
                std::move(names), std::move(description), true, { parameter<T>(std::move(param_name)) });
            return *this;
        }

        // adds option with a parameter and optional indication
        template<typename T>
        option_adder& operator()(
            std::string names, const std::string description, value_type<T>, std::string param_name, bool optional)
        {
            owner_.add_option<T>(
                std::move(names), std::move(description), optional, { parameter<T>(std::move(param_name)) });
            return *this;
        }

        // adds bare option
        option_adder& operator()(std::string names, const std::string description)
        {
            owner_.add_option(names, description);
            return *this;
        }

    private:
        command& owner_;
    };

public:
    // returns option adder for short addition semantics
    option_adder& add_options() { return *new option_adder(*this); }

    // add option method for more complex options
    // returns reference to the added option for the best control
    option& add_option(std::string names, const std::string description, bool optional = true)
    {
        options_.emplace_back(std::move(names), std::move(description), optional);
        return options_.back();
    }

    // overloading of previous method used to fully specify the option in one brief method call
    template<typename... T>
    option& add_option(
        std::string names, std::string description, bool optional, typename ::mbas::parameter_holder<T...>::type params)
    {
        options_.emplace_back(std::move(names), std::move(description), optional);

        auto& opt = options_.back();

        internal::add_param<0, mbas::parameter<T>...>(params, opt);

        return opt;
    }


    // returns documentation of the command
    std::string help()
    {
        std::stringstream help_str;

        help_str << "Options:" << std::endl;

        for (const auto& op : options_)
        {
            help_str << "\t";
            const auto& names = op.names;
            for (size_t i = 0; i < names.size(); i++)
            {
                const auto& name = names[i];
                help_str << (name.size() == 1 ? "-" : "--") << name;

                const auto& params = op.parameters();

                if (!params.empty())
                    help_str << (name.size() == 1 ? " " : "=");

                for (size_t j = 0; j < params.size(); ++j)
                {
                    const auto& param = params[j];
                    help_str << (param->mandatory ? "" : "[") << param->name << (param->mandatory ? "" : "]");
                    if (j != params.size() - 1)
                        help_str << " ";
                }

                if (i != names.size() - 1)
                    help_str << ", ";
            }

            help_str << std::endl << "\t\t";

            std::stringstream desc(op.description);

            size_t line_length = 0;
            while (!desc.eof())
            {
                if (line_length > 50)
                {
                    help_str << std::endl << "\t\t";
                    line_length = 0;
                }
                std::string tmp;
                desc >> tmp;
                help_str << tmp << " ";
                line_length += tmp.size() + 1;
            }

            help_str << std::endl << std::endl;
        }
        return help_str.str();
    }

    // parses the command line
    parsed_args parse(int argc, char** argv) const { return parser(this, argc, argv).parse(); }

private:
    std::vector<option> options_;

    const option* find_option(const std::string& option_name) const
    {
        for (auto&& it = options_.cbegin(); it != options_.cend(); ++it)
        {
            for (auto&& names_it = it->names.cbegin(); names_it != it->names.cend(); ++names_it)
            {
                if (*names_it == option_name)
                    return &*it;
            }
        }
        return nullptr;
    }

    bool check_mandatory_parsed(std::unordered_map<std::string, parsed_option*>& mapping) const
    {
        for (auto& opt : options_)
            if (!opt.optional && mapping.find(opt.names[0]) == mapping.end())
                return false;
        return true;
    }

    class parser
    {
        const command* cmd;
        std::unordered_map<std::string, parsed_option*> options_mapping;
        std::deque<parsed_option> options;
        std::vector<std::string> plain_args;

        bool parse_ok = true;
        bool delimiter_seen = false;

        int argc;
        char** argv;

    public:
        parser(const command* cmd, int argc, char** argv)
            : cmd(cmd)
            , argc(argc)
            , argv(argv)
        {}

        parsed_args parse()
        {
            for (size_t i = 1; i < (size_t)argc; ++i)
            {
                char* arg = argv[i];

                if (delimiter_seen)
                {
                    plain_args.push_back(arg);
                    continue;
                }

                if (arg[0] == '-')
                    parse_option(i);
                else
                    plain_args.push_back(arg);
            }

            if (!cmd->check_mandatory_parsed(options_mapping))
                parse_ok = false;

            return parsed_args(std::move(options_mapping), std::move(options), std::move(plain_args), parse_ok);
        }

        void parse_option(size_t& i)
        {
            char* arg = argv[i];

            std::string option_name;
            std::string first_par;
            bool first_par_parsed = false;
            bool option_parse_ok = true;
            std::vector<std::unique_ptr<parsed_param_base>> opt_params;
            const option* opt = nullptr;
            size_t parsed_par_count = 0;

            if (arg[1] == '-')
            { // starts with 2 dashes - long option
                if (arg[2] == '\0')
                {
                    delimiter_seen = true;
                    return;
                }

                first_par_parsed = parse_long_option(arg + 2, option_name, first_par);
                opt = cmd->find_option(option_name);
            }
            else
            { // starts with only 1 dash - short option
                if (arg[1] == '\0')
                {
                    parse_ok = false;
                    return;
                }

                first_par_parsed = parse_short_options(arg, opt, option_name, first_par);
            }

            bool option_constraint_ok = true;

            if (first_par_parsed)
            {
                if (!opt || opt->parameters().size() == 0)
                {
                    // a parameter was parsed, with the option name, but either the option is unknown or known flag
                    // option
                    auto par = std::make_unique<parsed_param<std::string>>(std::move(first_par), "", true, true);
                    opt_params.push_back(std::move(par));
                    options.emplace_back(option_name, std::move(opt_params), false, true, false, opt == nullptr);
                    if (opt)
                        add_option_mapping(*opt, &options.back());
                    else
                        options_mapping.emplace(option_name, &options.back());
                    parse_ok = false;
                    return;
                }

                add_parameter(
                    *opt->parameters()[0], std::move(first_par), option_parse_ok, option_constraint_ok, opt_params);
                parsed_par_count = 1;
            }
            else if (!opt)
            {
                options.emplace_back(
                    option_name, std::vector<std::unique_ptr<parsed_param_base>> {}, false, true, true, true);
                options_mapping.emplace(option_name, &options.back());
                return;
            }

            // parse next input arguments, until there is possibility that the option takes them as parameters
            while (parsed_par_count < opt->parameters().size())
            {
                ++i;
                if (i >= (size_t)argc)
                    break;
                else if (argv[i][0] == '-')
                {
                    --i;
                    break;
                }
                else
                {
                    add_parameter(*opt->parameters()[parsed_par_count],
                        argv[i],
                        option_parse_ok,
                        option_constraint_ok,
                        opt_params);
                    ++parsed_par_count;
                }
            }

            // the check whether option has all mandatory arguments is done only based on
            // the number of parsed arguments
            bool count_ok = true;
            if (parsed_par_count < opt->mandatory_count())
            {
                option_parse_ok = false;
                count_ok = false;
            }

            options.emplace_back(
                option_name, std::move(opt_params), option_parse_ok, option_constraint_ok, count_ok, false);
            if (!option_parse_ok)
                parse_ok = false;
            if (!option_constraint_ok)
                parse_ok = false;
            add_option_mapping(*opt, &options.back());
        }

        bool parse_long_option(std::string option_input, std::string& option_name, std::string& first_par)
        {
            size_t eq_pos = option_input.find_first_of('=');

            if (eq_pos == std::string::npos)
            {
                option_name = std::move(option_input);
                return false;
            }
            else
            {
                option_name = option_input.substr(0, eq_pos);
                first_par = option_input.substr(eq_pos + 1);
                return true;
            }
        }

        bool parse_short_options(char* arg, const option*& opt, std::string& option_name, std::string& first_par)
        {
            size_t i = 1;
            const option* prev_option = nullptr;
            std::string prev_name;
            option_name = arg[i];
            opt = cmd->find_option(option_name);
            ++i;

            while (opt && opt->parameters().size() == 0 && arg[i] != '\0')
            {
                prev_option = opt;
                prev_name = option_name;

                option_name = arg[i];
                opt = cmd->find_option(option_name);

                if (opt)
                {
                    options.emplace_back(
                        prev_name, std::vector<std::unique_ptr<parsed_param_base>> {}, true, true, true, false);
                    add_option_mapping(*prev_option, &options.back());
                }
                else
                {
                    opt = prev_option;
                    option_name = prev_name;
                    break;
                }
                ++i;
            }

            if (arg[i] != '\0')
            {
                first_par = std::string(arg + i);
                return true;
            }
            return false;
        }

        void add_parameter(const parameter_base& parameter,
            std::string value,
            bool& p_ok,
            bool& constraint_ok,
            std::vector<std::unique_ptr<parsed_param_base>>& opt_params) const
        {
            auto parsed_par = internal::parameter_base_accessor::parse(parameter, std::move(value));
            if (!parsed_par->parse_ok())
                p_ok = false;
            if (!parsed_par->constraint_ok())
                constraint_ok = false;
            opt_params.push_back(std::move(parsed_par));
        }

        void add_option_mapping(const option& opt, parsed_option* p_opt)
        {
            for (auto& name : opt.names)
                options_mapping.try_emplace(name, p_opt);
        }
    };
};

namespace internal {

std::unique_ptr<parsed_param_base> parameter_base_accessor::parse(
    const parameter_base& parameter, const std::string parsed)
{
    return parameter.parse(parsed);
}
} // namespace internal


} // namespace mbas

#endif