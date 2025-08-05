module ftorch_inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real64

   use cam_logfile,  only: iulog
 
   ! Import our library for interfacing with PyTorch
   use ftorch, only : torch_model, torch_tensor, torch_kCPU, torch_delete, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward

   implicit none

   ! Set working precision for reals
   integer, parameter :: wp = sp

   !integer, parameter :: kCPU = 0

   character(len=:), allocatable, save :: cb_torch_model 

   public ftorch_inference_cpu,init_ftorch_inference

   contains

   subroutine init_ftorch_inference()
        if (.not. allocated(cb_torch_model)) then
            cb_torch_model = "/global/homes/a/agett/python/ftorch/saved_simplenet_model_cpu.pt"
        end if
   end subroutine init_ftorch_inference

   subroutine ftorch_inference_cpu()

   ! Set up Fortran data structures
        real(wp), dimension(5), target :: in_data
        real(wp), dimension(5), target :: out_data

   ! Set up Torch data structures
   ! The net, a vector of input tensors (in this case we only have one), and the output tensor
        type(torch_model) :: model
        type(torch_tensor), dimension(1) :: in_tensors
        type(torch_tensor), dimension(1) :: out_tensors

   ! Optional: defensive check
        if (.not. allocated(cb_torch_model)) call init_ftorch_inference()

        write(iulog, *) 'Torch model path:',  trim(cb_torch_model)
        write(iulog, *) 'torch_kCPU (device):',  torch_kCPU

   ! Load ML model
   !     call torch_model_load(model, trim(cb_torch_model), torch_kCPU)

   ! Initialise data
        in_data = [0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]

   ! Create Torch input/output tensors from the above arrays
   !     call torch_tensor_from_array(in_tensors(1), in_data, torch_kCPU)
   !    call torch_tensor_from_array(out_tensors(1), out_data, torch_kCPU)

   ! Inference
   !     call torch_model_forward(model, in_tensors, out_tensors)

   ! Cleanup
   !     call torch_delete(model)
   !     call torch_delete(in_tensors)
   !    call torch_delete(out_tensors)

   end subroutine ftorch_inference_cpu


end module ftorch_inference