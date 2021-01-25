#include "SPECK3D_Compressor.h"

#include <cassert>

#ifdef USE_ZSTD
    #include "zstd.h"
#endif

SPECK3D_Compressor::SPECK3D_Compressor( size_t x, size_t y, size_t z )
                  : m_total_vals( x * y * z )
{
    m_cdf.set_dims( x, y, z );
    m_encoder.set_dims( x, y, z );
}

template< typename T >
auto SPECK3D_Compressor::copy_data( const T* p, size_t len ) -> RTNType
{
    if( len != m_total_vals )
        return RTNType::WrongSize;

    static_assert(std::is_floating_point<T>::value,
                  "!! Only floating point values are supported !!");

    m_val_buf = speck::unique_malloc<double>( len );
    for( size_t i = 0; i < len; i++ )
        m_val_buf[i] = p[i];

    return RTNType::Good;
}
template auto SPECK3D_Compressor::copy_data( const double*, size_t ) -> RTNType;
template auto SPECK3D_Compressor::copy_data( const float*,  size_t ) -> RTNType;


auto SPECK3D_Compressor::take_data( speck::buffer_type_d buf, size_t len ) -> RTNType
{
    if( len != m_total_vals )
        return RTNType::WrongSize;

    m_val_buf = std::move( buf );

    return RTNType::Good;
}
 

auto SPECK3D_Compressor::read_floats( const char* filename ) -> RTNType
{
    std::FILE* file = std::fopen( filename, "rb" );
    if( !file )
        return RTNType::IOError;

    std::fseek( file, 0, SEEK_END );
    const size_t file_size = std::ftell( file );
    std::fseek( file, 0, SEEK_SET );
    if( file_size % 4 != 0 || file_size / 4 != m_total_vals ) {
        std::fclose( file );
        return RTNType::WrongSize;
    }

    auto tmp_buf = speck::unique_malloc<float>(   file_size / 4 );
    size_t nread  = std::fread( tmp_buf.get(), 4, file_size / 4, file );
    std::fclose( file );
    if( nread != file_size / 4)
        return RTNType::IOError;
    
    return( this->copy_data( tmp_buf.get(), file_size / 4 ) );
}


auto SPECK3D_Compressor::compress() -> RTNType
{
    if( m_val_buf == nullptr )
        return RTNType::Error;

    m_cdf.take_data( std::move(m_val_buf), m_total_vals );
    m_val_buf = nullptr; // give the moved-from object a specified state
    m_cdf.dwt3d();
    auto cdf_out = m_cdf.release_data();
    if( cdf_out.first == nullptr || cdf_out.second != m_total_vals )
        return RTNType::Error;

    m_encoder.set_image_mean( m_cdf.get_mean() );
    m_encoder.take_data( std::move(cdf_out.first), cdf_out.second );

#ifdef QZ_TERM
    m_encoder.set_quantization_term_level( m_qz_lev );
#else
    m_encoder.set_bit_budget( size_t(m_bpp * m_total_vals) );
#endif

    return (m_encoder.encode());
}


auto SPECK3D_Compressor::get_encoded_bitstream() const
                         -> std::pair<speck::buffer_type_uint8, size_t>
{
    // After receiving the bitstream from SPECK3D, this method does 3 things:
    // 1) prepend a proper header containing meta data.
    // 2) potentially append a block of data that performs outlier correction.
    // 3) potentially apply ZSTD on the entire memory block except the meta data.

    auto speck_stream = m_encoder.get_encoded_bitstream();
    if( speck_stream.first == nullptr || speck_stream.second == 0 )
        return speck_stream;

    // Meta data definition:
    // the 1st byte records the current major version of SPECK, and
    // the 2nd byte records 8 booleans, with their meanings documented below:
    // 
    // bool_byte[0]  : if the rest of the stream is zstd compressed.
    // bool_byte[1]  : if this bitstream is for 3D (true) or 2D (false) data.
    // bool_byte[2]  : if there is error-bound data after the SPECK stream.
    // bool_byte[3-7]: unused 
    //
    uint8_t meta[2] = {uint8_t(SPECK_VERSION_MAJOR), 0};
    assert( sizeof(meta) == m_meta_size );
    bool metabool[8];
#ifdef USE_ZSTD
    metabool[0] = true;
#else
    metabool[0] = false;
#endif
    metabool[1] = true;
    for( int i = 2; i < 8; i++ )
        metabool[i] = false;
    speck::pack_8_booleans( meta[1], metabool );

#ifdef USE_ZSTD
    const size_t uncomp_size  = speck_stream.second;
    const size_t comp_buf_len = ZSTD_compressBound( uncomp_size );
    auto comp_buf = speck::unique_malloc<uint8_t>( m_meta_size + comp_buf_len );
    std::memcpy( comp_buf.get(), meta, m_meta_size );   // copy over metadata

    size_t comp_size = ZSTD_compress( comp_buf.get() + m_meta_size, comp_buf_len,
                                      speck_stream.first.get(), uncomp_size,
                                      ZSTD_CLEVEL_DEFAULT + 3 );
    if( ZSTD_isError( comp_size ))
        return {nullptr, 0};
    else
        return {std::move(comp_buf), m_meta_size + comp_size};   
#else
    const size_t total_size = m_meta_size + speck_stream.second;
    auto buf = speck::unique_malloc<uint8_t>( total_size );
    std::memcpy( buf.get(), meta, m_meta_size );  // copy over metadata
    std::memcpy( buf.get() + m_meta_size,         // copy over speck stream
                 speck_stream.first.get(), speck_stream.second );
    return {std::move(buf), total_size};
#endif
}
    

auto SPECK3D_Compressor::write_bitstream( const char* filename ) const -> RTNType
{
    auto stream = this->get_encoded_bitstream();
    if( stream.first == nullptr || stream.second == 0 )
        return RTNType::Error;

    return speck::write_n_bytes( filename, stream.second, stream.first.get() );
}


#ifdef QZ_TERM
void SPECK3D_Compressor::set_qz_level( int32_t q )
{
    m_qz_lev = q;
}
#else
auto SPECK3D_Compressor::set_bpp( float bpp ) -> RTNType
{
    if( bpp < 0.0 || bpp > 64.0 )
        return RTNType::InvalidParam;
    else {
        m_bpp = bpp;
        return RTNType::Good;
    }
}
#endif
