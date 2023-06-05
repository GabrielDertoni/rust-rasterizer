extern crate proc_macro;

use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Field, Fields, Meta};

#[proc_macro_derive(IntoSimd)]
pub fn into_simd_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let vis = input.vis;
    if input.generics.lt_token.is_some() || input.generics.where_clause.is_some() {
        return quote!(compile_error!(
            "generics and where clauses are note supported"
        ))
        .into();
    }

    let Data::Struct(structure) = input.data else {
        return quote!(compile_error!("only structs are supported")).into();
    };

    let ident = input.ident;
    let simd_struct = quote::format_ident!("{ident}Simd");

    let (struct_decl, struct_init, structure_of_array, constraints) = match structure.fields {
        Fields::Named(fields) => {
            let field_decls = fields.named.iter().map(|field| {
                let name = &field.ident;
                let ty = &field.ty;
                let vis = &field.vis;
                quote!(#vis #name: <#ty as IntoSimd>::Simd<LANES>)
            });
            let field_names = fields.named.iter().map(|field| &field.ident);
            let field_names2 = field_names.clone();
            let field_names3 = field_names.clone();
            let fields_init = field_names
                .clone()
                .map(|field| quote!(IntoSimd::splat(self.#field)));
            let fields_from_array = field_names
                .clone()
                .map(|field| quote!(StructureOfArray::from_array(array.map(|el| el.#field))));
            let constraints = fields.named.iter().map(|field| {
                let ty = &field.ty;
                quote!(#ty: IntoSimd)
            });
            let constraints2 = constraints.clone();
            let trait_constraints = constraints.clone();
            (
                quote! {
                    #vis struct #simd_struct<const LANES: usize>
                    where
                        std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount,
                        #(#constraints,)*
                    {
                        #(#field_decls,)*
                    }
                },
                quote! {
                    #simd_struct {
                        #(#field_names: #fields_init),*
                    }
                },
                quote! {
                    impl<const LANES: usize> StructureOfArray<LANES> for #simd_struct<LANES>
                    where
                        std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount,
                        #(#constraints2,)*
                    {
                        type Structure = #ident;

                        #[inline]
                        fn from_array(array: [Self::Structure; LANES]) -> Self {
                            #simd_struct {
                                #(#field_names2: #fields_from_array),*
                            }
                        }

                        fn index(&self, i: usize) -> Self::Structure {
                            #ident {
                                #(#field_names3: StructureOfArray::index(&self.#field_names3, i)),*
                            }
                        }
                    }
                },
                quote!(where #(#trait_constraints,)*),
            )
        }
        Fields::Unnamed(fields) => {
            let field_decls = fields.unnamed.iter().map(|field| {
                let ty = &field.ty;
                let vis = &field.vis;
                quote!(#vis <#ty as IntoSimd>::Simd<LANES>)
            });
            let fields_init =
                (0..fields.unnamed.len()).map(|field| quote!(IntoSimd::splat(self.#field)));
            let fields_from_array = (0..fields.unnamed.len())
                .map(|field| quote!(StructureOfArray::from_array(array.map(|el| el.#field))));
            let constraints = fields.unnamed.iter().map(|field| {
                let ty = &field.ty;
                quote!(#ty: IntoSimd)
            });
            let trait_constraints = constraints.clone();
            let _from_array = quote! {
                #simd_struct(#(#fields_from_array),*)
            };
            (
                quote! {
                    #vis struct #simd_struct<const LANES: usize>(#(#field_decls),*)
                    where
                        std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount,
                        #(#constraints),*;
                },
                quote! {
                    #vis struct #simd_struct(#(#fields_init,)*);
                },
                quote!(compile_error!("todo")), // TODO
                quote!(where #(#trait_constraints,)*),
            )
        }
        Fields::Unit => (
            quote!(#(#vis) struct #simd_struct;),
            quote!(#simd_struct),
            quote!(compile_error!("todo")), // TODO
            quote!(),
        ),
    };

    quote! {
        #struct_decl

        impl IntoSimd for #ident
        #constraints
        {
            type Simd<const LANES: usize> = #simd_struct<LANES>
            where
                std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount;

            #[inline]
            fn splat<const LANES: usize>(self) -> Self::Simd<LANES>
            where
                std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount,
            {
                #struct_init
            }
        }

        #structure_of_array
    }
    .into()
}

#[proc_macro_derive(Attributes, attributes(position))]
pub fn attributes_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    if input.generics.lt_token.is_some() || input.generics.where_clause.is_some() {
        return quote!(compile_error!(
            "generics and where clauses are note supported"
        ))
        .into();
    }

    let Data::Struct(structure) = input.data else {
        return quote!(compile_error!("only structs are supported")).into();
    };

    let ident = input.ident;

    let (position_field, interpolate) = match structure.fields {
        Fields::Named(fields) => {
            let mut it = fields.named.iter();
            let find_closure = |field: &&Field| -> bool {
                field
                    .attrs
                    .iter()
                    .find(|attr| {
                        let Meta::Path(path) = &attr.meta else { return false };
                        path.is_ident("position")
                    })
                    .is_some()
            };
            let Some(position_field) = it.find(&find_closure) else {
                return quote!(compile_error!(
                    "no field is marked `#[position]`, please choose one appropriate field for the position attribute"
                )).into();
            };
            if it.find(find_closure).is_some() {
                return quote!(compile_error!(
                    "two fields marked `#[position]` were found, please pick only one"
                ))
                .into();
            }

            let field_names = fields.named.iter().map(|field| &field.ident);

            let interpolate = quote! {
                Self::Simd {
                    #(#field_names: w.x * p0.#field_names.splat() + w.y * p1.#field_names.splat() + w.z * p2.#field_names.splat()),*
                }
            };

            (position_field.ident.clone(), interpolate)
        }
        Fields::Unnamed(_) => todo!(),
        Fields::Unit => todo!(),
    };

    quote! {
        impl Attributes for #ident {
            #[inline]
            fn interpolate<const LANES: usize>(
                p0: &Self,
                p1: &Self,
                p2: &Self,
                w: Vec<Simd<f32, LANES>, 3>,
            ) -> Self::Simd<LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                #interpolate
            }

            #[inline(always)]
            fn position(&self) -> &Vec4 {
                &self.#position_field
            }

            #[inline(always)]
            fn position_mut(&mut self) -> &mut Vec4 {
                &mut self.#position_field
            }
        }
    }
    .into()
}
