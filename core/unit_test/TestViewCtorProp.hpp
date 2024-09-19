//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace Test {

// Check Kokkos::Impl::is_view_label.
TEST(TEST_CATEGORY, is_view_label) {
  static_assert(Kokkos::Impl::is_view_label<std::string>::value);

  constexpr unsigned N = 3;
  static_assert(Kokkos::Impl::is_view_label<const char[N]>::value);
  static_assert(Kokkos::Impl::is_view_label<char[N]>::value);

  static_assert(!Kokkos::Impl::is_view_label<char*>::value);
}

// Check traits of Kokkos::Impl::ViewCtorProp<void, std::string>.
TEST(TEST_CATEGORY, vcp_label_base_traits) {
  using vcp_label_base_t = Kokkos::Impl::ViewCtorProp<void, std::string>;

  static_assert(std::is_default_constructible_v<vcp_label_base_t>);
  static_assert(std::is_copy_constructible_v<vcp_label_base_t>);
  static_assert(std::is_copy_assignable_v<vcp_label_base_t>);
  static_assert(std::is_move_constructible_v<vcp_label_base_t>);
  static_assert(std::is_move_assignable_v<vcp_label_base_t>);

  static_assert(std::is_same_v<typename vcp_label_base_t::type, std::string>);

  static_assert(std::is_constructible_v<vcp_label_base_t, std::string>);
  static_assert(std::is_constructible_v<vcp_label_base_t, const std::string&>);
  static_assert(std::is_constructible_v<vcp_label_base_t, std::string&&>);

  constexpr unsigned N = 3;
  static_assert(std::is_constructible_v<vcp_label_base_t, const char[N]>);
  static_assert(std::is_constructible_v<vcp_label_base_t, char*>);
}

// Check traits of Kokkos::Impl::ViewCtorProp<std::string>.
TEST(TEST_CATEGORY, vcp_label_traits) {
  using vcp_label_base_t = Kokkos::Impl::ViewCtorProp<void, std::string>;
  using vcp_label_t      = Kokkos::Impl::ViewCtorProp<std::string>;

  static_assert(std::is_base_of_v<vcp_label_base_t, vcp_label_t>);

  static_assert(vcp_label_t::has_label);

  static_assert(std::is_default_constructible_v<vcp_label_t>);
  static_assert(std::is_copy_constructible_v<vcp_label_t>);
  static_assert(std::is_copy_assignable_v<vcp_label_t>);
  static_assert(std::is_move_constructible_v<vcp_label_t>);
  static_assert(std::is_move_assignable_v<vcp_label_t>);

  static_assert(std::is_constructible_v<vcp_label_t, std::string>);
  static_assert(std::is_constructible_v<vcp_label_t, const std::string&>);
  static_assert(std::is_constructible_v<vcp_label_t, std::string&&>);

  constexpr unsigned N = 3;
  static_assert(std::is_constructible_v<vcp_label_t, const char[N]>);
  static_assert(std::is_constructible_v<vcp_label_t, char*>);
}

// Check functions related to Kokkos::Impl::ViewCtorProp<std::string>.
TEST(TEST_CATEGORY, vcp_label_fcns) {
  using vcp_label_t = Kokkos::Impl::ViewCtorProp<std::string>;

  // Check that the user-declared constructor can move.
  std::string label("our label");

  vcp_label_t prop(std::move(label));

  ASSERT_TRUE(label.empty());
  ASSERT_EQ(Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop),
            "our label");

  // Check the user-declared copy onstructor.
  vcp_label_t prop_copy_constr(prop);
  ASSERT_EQ(Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop),
            "our label");
  ASSERT_EQ(
      Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop_copy_constr),
      "our label");

  // Check the compiler-generated copy assignment operator.
  vcp_label_t prop_copy_asgmt;
  prop_copy_asgmt = prop;
  ASSERT_EQ(Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop),
            "our label");
  ASSERT_EQ(Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop_copy_asgmt),
            "our label");

  // Check the user-declared move constructor.
  vcp_label_t prop_move_constr(std::move(prop));
  ASSERT_TRUE(Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop).empty());
  ASSERT_EQ(
      Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop_move_constr),
      "our label");

  // Check the compiler-generated move assignment operator.
  vcp_label_t prop_move_asgmt;
  prop_move_asgmt = std::move(prop_copy_asgmt);
  ASSERT_TRUE(
      Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop_copy_asgmt)
          .empty());
  ASSERT_EQ(Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop_move_asgmt),
            "our label");

  // Check Kokkos::Impl::with_properties_if_unset.
  auto prop_empty = Kokkos::view_alloc();

  static_assert(!decltype(prop_empty)::has_label);

  const auto prop_with_label = Kokkos::Impl::with_properties_if_unset(
      prop_empty, std::string("our label"));
  static_assert(decltype(prop_with_label)::has_label);
  ASSERT_EQ(Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(prop_with_label),
            "our label");
}

}  // namespace Test
