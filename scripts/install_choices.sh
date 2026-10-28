# Guided-installer answers. Sourced by install_pixvqa.sh.
# pixvqa_choice NAME prints yes, no, unset, or invalid.
# An unset variable keeps the interactive prompt. 1/y/yes and 0/n/no do not.
pixvqa_choice() {
    local name="$1" value
    [[ -v $name ]] || { printf 'unset\n'; return 0; }
    value="${!name}"
    case "${value,,}" in
        1|y|yes) printf 'yes\n' ;;
        0|n|no) printf 'no\n' ;;
        *) printf 'invalid\n' ;;
    esac
}
